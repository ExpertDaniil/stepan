"""Small CPU tabular networks, trained from scratch inside the worker lease."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from agent.feature_engineering import _keys


class TabularNetwork(BaseEstimator):
    def __init__(self, regression=False, parameters=None, cpu_threads=1, seed=42, fit_context=None, classes=None):
        self.regression, self.parameters, self.cpu_threads, self.seed = regression, parameters, cpu_threads, seed
        self.fit_context, self.classes = fit_context, classes

    def _prepare(self, X, fit=False):
        if fit:
            self.columns_ = list(X)
            self.numeric_ = [c for c in X if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
            self.categorical_ = [c for c in X if c not in self.numeric_]
            self.category_maps_ = {}
            for c in self.categorical_:
                counts = _keys(X[c]).value_counts()
                self.category_maps_[c] = {key:i+1 for i,key in enumerate(counts.index[:4096])}
        if set(X) != set(self.columns_):
            raise ValueError('Neural input column set changed')
        numeric = X[self.numeric_].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        numeric[~np.isfinite(numeric)] = np.nan
        missing = np.isnan(numeric).astype(np.float32)
        if fit:
            self.medians_ = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in numeric.T], dtype=np.float32)
        numeric = np.where(np.isnan(numeric), self.medians_, numeric)
        if fit:
            self.means_, self.scales_ = numeric.mean(axis=0), np.maximum(numeric.std(axis=0), 1e-6)
        numeric = np.clip((numeric-self.means_)/self.scales_, -20, 20).astype(np.float32)
        categorical = np.column_stack([_keys(X[c]).map(self.category_maps_[c]).fillna(0).to_numpy(dtype=np.int64)
                                        for c in self.categorical_]) if self.categorical_ else np.empty((len(X),0), dtype=np.int64)
        return numeric, missing, categorical

    def _network(self, outputs):
        import torch
        from torch import nn
        options = dict(self.parameters or {})
        width, blocks = options.get('width',32), options.get('blocks',2)
        kind = options.get('architecture','transformer')
        dropout = options.get('dropout',.1)
        numeric = len(self.numeric_)
        category_sizes = [len(self.category_maps_[c])+1 for c in self.categorical_]
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(numeric,width)*.02)
                self.bias = nn.Parameter(torch.zeros(numeric,width))
                self.missing = nn.Parameter(torch.randn(numeric,width)*.02)
                self.embeddings = nn.ModuleList([nn.Embedding(n,width) for n in category_sizes])
                self.cls = nn.Parameter(torch.zeros(1,1,width))
                if kind == 'transformer':
                    layer = nn.TransformerEncoderLayer(width,4,dim_feedforward=width*2,dropout=dropout,
                        batch_first=True,norm_first=True,activation='gelu')
                    self.body = nn.TransformerEncoder(layer,blocks,enable_nested_tensor=False)
                    self.head = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,outputs))
                else:
                    self.body = nn.Sequential(nn.Linear((numeric+len(category_sizes))*width,width*2),nn.ReLU(),
                        nn.Dropout(dropout),nn.Linear(width*2,width),nn.ReLU())
                    self.head = nn.Linear(width,outputs)

            def forward(self, x, mask, cats):
                pieces = [x[:,:,None]*self.weight+self.bias+mask[:,:,None]*self.missing] if numeric else []
                pieces += [embedding(cats[:,i])[:,None,:] for i,embedding in enumerate(self.embeddings)]
                tokens = torch.cat(pieces,dim=1)
                if kind == 'transformer':
                    tokens = torch.cat([self.cls.expand(len(x),-1,-1),tokens],dim=1)
                    return self.head(self.body(tokens)[:,0])
                return self.head(self.body(tokens.flatten(1)))
        return Network()

    def fit(self, X, y):
        import torch
        from agent.learned_features import inner_splits
        options = dict(self.parameters or {})
        if options.get('width',32) not in (16,32,64) or not 1 <= options.get('blocks',2) <= 3:
            raise ValueError('CPU neural size outside owned bounds')
        if options.get('architecture','transformer') not in ('transformer','mlp'):
            raise ValueError('Unknown neural architecture')
        torch.set_num_threads(max(1,self.cpu_threads))
        prepared = self._prepare(X,fit=True)
        tensors = [torch.from_numpy(v) for v in prepared]
        raw = np.asarray(y)
        if self.regression:
            raw = raw.astype(np.float32)
            self.target_mean_, self.target_scale_ = float(raw.mean()), max(1e-6,float(raw.std()))
            target = torch.from_numpy((raw-self.target_mean_)/self.target_scale_).reshape(-1,1)
            outputs=1
        else:
            self.classes_ = np.asarray(self.classes or sorted(set(raw.astype(str))),dtype=str)
            target = torch.tensor([list(self.classes_).index(str(v)) for v in raw],dtype=torch.long)
            outputs=len(self.classes_)
        context = self.fit_context or {}
        split = list(inner_splits(X,raw,classification=not self.regression,folds=4,seed=self.seed,
            time_column=context.get('column') if context.get('strategy')=='temporal_holdout' else None,
            group_column=context.get('column') if context.get('strategy')=='group_holdout' else None))
        train, valid = split[-1] if split else (np.arange(len(X)),np.array([],dtype=int))
        epochs, patience = int(options.get('epochs',12)), 3
        batch_size = int(options.get('batch_size',256))
        if not 1<=epochs<=80 or not 32<=batch_size<=1024:
            raise ValueError('CPU neural training outside owned bounds')
        loss_fn = torch.nn.MSELoss() if self.regression else torch.nn.CrossEntropyLoss()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            network = self._network(outputs)
            optimizer = torch.optim.AdamW(network.parameters(),lr=options.get('learning_rate',.001),weight_decay=options.get('weight_decay',.0001))
            best_loss,best_epoch,stale = float('inf'),epochs,0
            generator = np.random.RandomState(self.seed)
            for epoch in range(epochs):
                network.train()
                order = generator.permutation(train)
                for start in range(0,len(order),batch_size):
                    rows=order[start:start+batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss=loss_fn(network(*(v[rows] for v in tensors)),target[rows])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite neural loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.);optimizer.step()
                if len(valid):
                    network.eval()
                    with torch.no_grad():
                        total=sum(float(loss_fn(network(*(v[rows] for v in tensors)),target[rows]))*len(rows)
                            for rows in np.array_split(valid,max(1,int(np.ceil(len(valid)/batch_size)))))
                    value=total/len(valid)
                    if value < best_loss-1e-5:best_loss,best_epoch,stale=value,epoch+1,0
                    else:stale+=1
                    if stale>=patience:break
            # Early-stop labels belong only to this fit. Relearn all fit rows at
            # the chosen epoch count; no outer/confirmation labels are observed.
            torch.manual_seed(self.seed)
            network=self._network(outputs)
            optimizer=torch.optim.AdamW(network.parameters(),lr=options.get('learning_rate',.001),weight_decay=options.get('weight_decay',.0001))
            for _ in range(best_epoch):
                network.train();order=generator.permutation(len(X))
                for start in range(0,len(order),batch_size):
                    rows=order[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
                    loss=loss_fn(network(*(v[rows] for v in tensors)),target[rows])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite neural refit loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.);optimizer.step()
            self.state_={k:v.detach().cpu().numpy().copy() for k,v in network.state_dict().items()}
        self.outputs_,self.best_epoch_=outputs,best_epoch
        self.report_={'operator':'tabular_network','architecture':options.get('architecture','transformer'),
            'fit_rows':len(X),'early_stop_fit_rows':len(train),'early_stop_valid_rows':len(valid),
            'best_epoch':best_epoch,'weights':'from_scratch','history_linkage':'not_inferred',
            'preprocessing':'fit_only; unknown_category=0; native_missing_token; numerical_clipping=20_standard_deviations'}
        return self

    def _logits(self,X):
        import torch
        with torch.random.fork_rng(devices=[]):
            network=self._network(self.outputs_)
            network.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in self.state_.items()})
        network.eval();prepared=[torch.from_numpy(v) for v in self._prepare(X)]
        with torch.no_grad():
            result=[network(*(v[start:start+512] for v in prepared)).numpy() for start in range(0,len(X),512)]
        return np.concatenate(result) if result else np.empty((0,self.outputs_))

    def predict_proba(self,X):
        if self.regression:raise TypeError('Regression has no class probabilities')
        logits=self._logits(X);logits-=logits.max(axis=1,keepdims=True)
        values=np.exp(logits);return values/values.sum(axis=1,keepdims=True)

    def predict(self,X):
        if self.regression:return self._logits(X).reshape(-1)*self.target_scale_+self.target_mean_
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
