import polars as pl
from plotnine import *
from pyprojroot.here import here
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load data
brand_data = pl.read_csv(here('data/brand-ratings.csv'))

# Aggregate the ratings by brand
brand_means = (brand_data
  .group_by(pl.col('brand'))
  .agg(pl.all().exclude('brand').mean())
)

# Specify the data to model
X = brand_means.drop('brand')

# Create a pipeline
pca_pipe = Pipeline([
  ('normalize', StandardScaler()),
  ('pca', PCA(n_components = 2))
])

# Fit PCA and get scores
pca_fit = pca_pipe.fit(X)
pca_scores = pca_fit.transform(X)
pca_steps = pca_fit.named_steps['pca']

# Create a brand map
brand_map = pl.DataFrame({
  'brand': brand_means.get_column('brand'),
  'PC1': pca_scores[:, 0],
  'PC2': pca_scores[:, 1]
})

# Plot the brand map
(ggplot(brand_map, aes(
    x = 'PC1', y = 'PC2', label = 'brand'
  ))
  + geom_hline(yintercept = 0, 
    linetype = 'dashed', 
    color = 'grey'
  )
  + geom_vline(xintercept = 0, 
    linetype = 'dashed', 
    color = 'grey'
  )
  + geom_point(size = 3)
  + geom_text(nudge_y = 0.1, va = 'bottom')
)

# Create a loadings data frame
pca_loadings = pl.DataFrame(
  pca_steps.components_[:2].T,
  schema=[f"PC{i}" for i in range(1, 3)],
)

# Print the first principal component
(pca_loadings
  .with_columns(variable = pl.Series(X.columns))
  .select('PC1', 'variable')
  .sort('PC1', descending = True)
)

# Print the second principal component
(pca_loadings
  .with_columns(variable = pl.Series(X.columns))
  .select('PC2', 'variable')
  .sort('PC2', descending = True)
)

