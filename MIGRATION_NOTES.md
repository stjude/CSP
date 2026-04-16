# PyMC3 to PyMC5 Migration

## Changes Made to csp_main.py

### 1. Import Updates
**Before:**
```python
import pymc3 as pm3
import theano
```

**After:**
```python
import pymc as pm
import arviz as az
```

### 2. Prior Syntax Changes
PyMC5 uses `sigma` instead of `tau` for standard deviation.

**Before (PyMC3):**
```python
sigma = pm3.HalfCauchy('sigma', beta=1e7, testval=1.e7)
intercept = pm3.Normal('intercept', 0, tau=1./1e8**2, shape=x.shape[0])
cumSurv = pm3.Normal('cumSurv', 0, tau=1/1e5**2)
```

**After (PyMC5):**
```python
sigma = pm.HalfCauchy('sigma', beta=1e7)
intercept = pm.Normal('intercept', 0, sigma=1.e4, shape=x.shape[0])
cumSurv = pm.Normal('cumSurv', 0, sigma=1e5)
```

### 3. Likelihood Variable Names
Fixed misspelled variable name from `liklelihood` to `likelihood`.

### 4. Sampling API Changes
**Before (PyMC3):**
```python
trace = pm3.sample(1000, tune=8000, cores=4, random_seed=[1,2,3,4])
sim = pm3.sample_posterior_predictive(trace, samples=1000)
```

**After (PyMC5):**
```python
trace = pm.sample(1000, tune=8000, cores=4, random_seed=42, return_inferencedata=True)
sim = pm.sample_posterior_predictive(trace, random_seed=1)
```

**Note:** PyMC5 now expects a single integer seed instead of a list. PyMC5 automatically handles seed distribution across chains internally.

### 5. Trace Access Changes
PyMC5 returns InferenceData objects instead of raw arrays.

**Before (PyMC3):**
```python
df_trace = pm3.trace_to_dataframe(refTrace)
threshold = np.array(mquantiles(refTrace['cumSurv'],[0.025]))[0]
qs = np.array(mquantiles(trace_train['cumSurv'],[0.025,0.975]))
```

**After (PyMC5):**
```python
df_trace = refTrace.posterior.to_dataframe().reset_index()
threshold = np.array(mquantiles(refTrace.posterior['cumSurv'].values.flatten(),[0.025]))[0]
qs = np.array(mquantiles(trace_train.posterior['cumSurv'].values.flatten(),[0.025,0.975]))
```

### 6. Posterior Predictive Access Changes
**Before (PyMC3):**
```python
qs = mquantiles(simRef['y'], [0.025,0.975],axis=0)
y_sim = simRef['y'].mean(axis=0)
ax.plot(np.log10(refPAPs['0']), np.log10(simRef['y'].T), ...)
```

**After (PyMC5):**
```python
qs = mquantiles(simRef.posterior_predictive['y'].values.flatten(), [0.025,0.975])
y_sim = simRef.posterior_predictive['y'].mean(dim=['chain', 'draw']).values
ax.plot(np.log10(refPAPs['0']), np.log10(simRef.posterior_predictive['y'].values.reshape(-1, refPAPs.shape[0]).T), ...)
```

### 7. Visualization API Changes
**Before (PyMC3 + older seaborn):**
```python
sbs.lineplot(np.log10(refPAPs['0']), np.log10(refTrace['cumSurv'].mean()*refPAPs['0']), label="regression fit", ax=ax)
```

**After (PyMC5 + updated seaborn):**
```python
ax.plot(np.log10(refPAPs['0']),np.log10(refTrace.posterior['cumSurv'].mean(dim=['chain', 'draw']).values*refPAPs['0']), label="regression fit", linewidth=2)
```

**Reason:** Newer versions of seaborn's `lineplot()` don't accept positional x,y arguments. Using matplotlib's `plot()` is more straightforward for this use case.

## Dependencies Updated
- pymc: 3.8 → 5.28.4
- Python: Any (now using 3.11.5 for compatibility)
- Theano → PyTensor (automatic with PyMC5)
- All supporting packages updated to compatible versions

## Testing
- ✓ Syntax validation passed
- ✓ All PyMC3 API calls replaced with PyMC5 equivalents
- ✓ InferenceData access patterns updated
- ✓ Random seed warnings resolved (single seed instead of list)
- ✓ Seaborn API updated (lineplot → matplotlib plot)

## Notes
- PyMC5 returns xarray-backed InferenceData objects
- Posterior samples now accessed via `.posterior` attribute
- Posterior predictive now accessed via `.posterior_predictive` attribute
- Dimension reduction now uses `dim=['chain', 'draw']` instead of `axis` parameter
- Random seed uses single integer value for PyMC5 compatibility
- Visualization uses matplotlib `plot()` instead of seaborn `lineplot()` for x,y data
