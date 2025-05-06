import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
    """
    Conjugate Gradient solver from:
    https://github.com/aravindr93/imaml_dev/blob/master/implicit_maml/utils.py
    
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """
       
    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x





class BiImpute:
    def __init__(self):
        self.moving_dot_product = None
        

    @staticmethod
    def flatten_tuple(tup):
        return torch.cat([t.contiguous().view(-1) for t in tup])

    @staticmethod
    def unflatten(flat, params):
        out = []
        offset = 0
        for p in params:
            numel = p.numel()
            out.append(flat[offset:offset+numel].view_as(p))
            offset += numel
        return out

    def cache_grad(self, x_train, m_train):
        
        # 1) form train‐imputation
        x_imp = self.teacher.forward_impute(x_train, m_train, training=True)
        x_imp = x_imp.transpose(0, 1)   # (batch, seq, feats)

        # 2) compute student’s training loss
        #    here we predict the last step from all but last

        S, T, N = x_imp.shape
        losses = []
        
        for t in range(T-1):
            y_pred = self.student(x_imp[:, t:T-1], True, self.student_dropout)
            y_true = x_imp[:, T-1, :]
            losses.append(
                ((y_pred[:, :] - y_true) ** 2) # * m_train[:, -1, :]
            )
                
        loss = torch.stack(losses).mean()

        # 3) grad L_train wrt student params
        grad_params = torch.autograd.grad(loss, self.student.parameters(),
                                          create_graph=True,
                                          retain_graph=True,)
        self.flat_grad = self.flatten_tuple(grad_params)

    def hessian_vector_product(self, vector, x_train, m_train):
        flat_grad = self.flat_grad
        
        # 4) form (flat_grad)ᵀ vector
        dot = torch.dot(flat_grad, vector)

        # 5) backprop that to get hvp in student‐param space
        hvp = torch.autograd.grad(dot, self.student.parameters(),
                                  retain_graph=True,)
        flat_hvp = self.flatten_tuple(hvp)

        return flat_hvp

    def matrix_evaluator(self, lam):
        """
        Returns a function f(v) = (H + lam I) v,
        ready to be plugged into cg_solve.
        """
        def evaluator(v):
            # v is a flat 1-D tensor
            hvp = self.hessian_vector_product(v, 
                         x_train=self._latest_t_batch_bi,   # see note below
                         m_train=self._latest_t_mask_bi)
            return hvp + lam * v
        return evaluator


    def step_fn(self, args, models, optimizers, all_samples):

        teacher, student = models
        teacher_optimizer, student_optimizer = optimizers

        self.teacher = teacher
        self.student = student
        
        
        teacher.train().to(args.device)
        student.train().to(args.device)
        
        (t_batch, t_mask), (t_batch_bi, t_mask_bi), (v_batch, v_mask) = all_samples
        
        self._latest_t_batch_bi = t_batch_bi
        self._latest_t_mask_bi  = t_mask_bi
        
        t_batch_imputed = teacher.forward_impute(t_batch_bi, t_mask_bi, training=True)
        v_batch_imputed = teacher.forward_impute(v_batch, v_mask, training=True)
        
        t_batch_imputed = t_batch_imputed.transpose(0, 1)
        v_batch_imputed = v_batch_imputed.transpose(0, 1)

        t_mask_bi_transposed = t_mask_bi.transpose(0, 1)
        v_mask_transposed = v_mask.transpose(0, 1)

        
        preds = {}
        labels = {}
        nlll = {}

        S, T, N = t_batch_imputed.shape
        train_losses = []
        val_old_losses = []

        self.student_dropout = args.student_dropout
        for t in range(T-1):
            preds['s_on_t'] = student(t_batch_imputed[:, t:T-1], True, self.student_dropout)
            train_losses.append(
                ((preds['s_on_t'][:, :] - t_batch_imputed[:, -1, :]) ** 2) # * t_mask_bi_transposed[:, -1, :]
            )

        
        for t in range(T-1):
            preds['s_on_v_old'] = student(v_batch_imputed[:, t:T-1] * v_mask_transposed[:, t:T-1], False)
            val_old_losses.append(
                (((preds['s_on_v_old'][:, :] - v_batch_imputed[:, -1, :]) ** 2) * v_mask_transposed[:, -1, :]).sum() /
                v_mask_transposed[:, -1, :].sum()
            )

        train_loss = torch.stack(train_losses).mean()
        val_old   = torch.stack(val_old_losses).mean()

        
        nlll['s_on_t'] = train_loss
        nlll['s_on_v_old'] = val_old
        
        student_optimizer.zero_grad()
        nlll['s_on_t'].backward(retain_graph=True)
        student_optimizer.step()


        S, T, N = t_batch_imputed.shape
        train_new_losses = []
        val_new_losses = []
        
        for t in range(T-1):
            preds['s_on_t_new'] = student(t_batch_imputed[:, t:T-1], True, self.student_dropout)
            train_new_losses.append(
                ((preds['s_on_t_new'][:, :] - t_batch_imputed[:, -1, :]) ** 2) # * t_mask_bi_transposed[:, -1, :]
            )

        
        for t in range(T-1):
            preds['s_on_v_new'] = student(v_batch_imputed[:, t:T-1] * v_mask_transposed[:, t:T-1], False)
            val_new_losses.append(
                (((preds['s_on_v_new'][:, :] - v_batch_imputed[:, -1, :]) ** 2) * v_mask_transposed[:, -1, :]).sum() /
                v_mask_transposed[:, -1, :].sum()
            )
        
        train_new = torch.stack(train_new_losses).mean()
        val_new   = torch.stack(val_new_losses).mean()
        
        nlll['s_on_t_new'] = train_new
        nlll['s_on_v_new'] = val_new
        
        
        teacher_optimizer.zero_grad()  
        
        
        v_tuple = torch.autograd.grad(
            outputs=nlll['s_on_v_new'],
            inputs=student.parameters(),
            create_graph=True,
            retain_graph=True,
        )
        
        v_flat = torch.cat([g.contiguous().view(-1) for g in v_tuple])
        
        self.cache_grad( x_train=self._latest_t_batch_bi,
                         m_train=self._latest_t_mask_bi)
        A = self.matrix_evaluator(lam=args.student_lambda)
        
        u_flat = cg_solve(A, v_flat, cg_iters=10)
        
        u_tuple = self.unflatten(u_flat, list(student.parameters()))

        u_flat_det = u_flat.detach()

        g_tuple = torch.autograd.grad(
            outputs=nlll['s_on_t_new'],
            inputs=student.parameters(),
            create_graph=True,
            retain_graph=True
        )
        g_flat = torch.cat([g.contiguous().view(-1) for g in g_tuple])
        
        dot_product = torch.dot(g_flat, u_flat_det)
        
        implicit = torch.autograd.grad(
            outputs=dot_product,
            inputs=teacher.parameters(),
            retain_graph=False,
            allow_unused=True,
        )

        # zero fill the grads so I don't have to deal with None grads
        implicit_filled = []
        for ig, p in zip(implicit, teacher.parameters()):
            if ig is None:
                implicit_filled.append(torch.zeros_like(p))
            else:
                implicit_filled.append(ig)


        teacher_optimizer.zero_grad()

        for p, ig in zip(teacher.parameters(), implicit_filled):
            p.grad = - ig

        ssl_loss = teacher.forward_ssl(t_batch, t_mask, mask_ratio=args.mask_ratio)
        ssl_loss.backward()

        teacher_optimizer.step()

        nlll['mae'] = ssl_loss
        
        
        logs = collections.OrderedDict()
        logs['nlll/student_on_t'] = nlll['s_on_t'].item()
        logs['nlll/student_on_v'] = nlll['s_on_v_new'].item()
        logs['mae'] = nlll['mae'].item()
        

        self.step_info = logs