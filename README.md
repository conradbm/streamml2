<h1>Streamlined Machine Learning</h1>
<hr>
<strong>Streamlined Machine Learning</strong> is best explained by describing it's structure. There exist three primary functions: 1. Transformation Preprocessing 2. Model Selection 3. Feature Selection. <em>streamml</em> contains a set of robust functions built on top of the <em>sklearn</em> framework and aims at streamlining all of the processes in the context of a flow.

<h2> Background </h2>
The three main classes in the <em>streamml</em> ecosystem are: TransformationStream, ModelSelectionStream, and FeatureSelectionStream. The underlying assumption before running any of these objects and their capabilities is that you have cleaned and completely preprocessed your data of all the nasty you would do before running any model or transformation in <em>sklearn</em>. All X and y data piped into these models must be a <em>pandas.DataFrame</em>, even your <em>pandas.Series</em> style y data (this simplifies and unifies functionality accross the ecosystem). That said, <strong>TransformationStream</strong> is constructed with X, then has the ability to flow through a cadre of different manifold, clustering, or transformation functions built into the ecosystem (which are explained the documentation in further detail). <strong>ModelSelectionStream</strong> is constructed with both X and y, where y can be categorical (binary or n-ary), then has the ability to flow through a cadre of different <em>sklearn</em> based model objects. The underlying assumption is that your y data has been categorized into a numeric representation, as this is how <em>sklearn</em> prefers it. We recommend you simply use <em>pandas.factorize</em> to accomplish this, but this is not done explicitely or implicitely for you.  Lastly <strong>FeatureSelectionStream</strong> is constructed with both X and y, then has the ability to flow through a cadre of very specific types of model objects and ensemble functions. The models are ones in the <em>sklearn</em> packages that contain <code> coef_</code>, <code>feature_importance_</code>, or <code>p-values</code> attributes produced after the hyper-tuning phase or running the model. As this is unintuitive at first, these include, but are not limited to: OLS p-values, Random Forest feature importance, or Lasso coefficients.  

<h2>Lets Get Started </h2>
In the <em>streamml</em> ecosystem, as mentioned above, we must build a stream object. The idea is within this stream, we can flow through very specific objects that are optimized for us behind the scenes. Yup. That's it. All of the gridsearching and pipelining procedures you are use to doing everytime you see a dataset are already built in. Just construct a stream and then <code>.flow([...])</code> right on through it, and it will return your hypertuned models, transformed feature-space, or a subspace of features that are most pronounced within your data. 
Streaming Capabilities provided:
<ul>
<li><code>TransformationStream</code>, meant to flow through preprocessing techniques such as: scaling, normalizing, boxcox, binarization, pca, or kmeans aimed at returning a desired input dataset for model development.</li>

<li><code>ModelSelectionStream</code>. 
<p>Regression Models:
{"lr" : linearRegression,
"svr" : supportVectorRegression,
"rfr":randomForestRegression,
"abr":adaptiveBoostingRegression,
"knnr":knnRegression,
"ridge":ridgeRegression,
"lasso":lassoRegression,
"enet":elasticNetRegression,
"mlpr":multilayerPerceptronRegression,
"br":baggingRegression,
"dtr":decisionTreeRegression,
"gbr":gradientBoostingRegression,
"gpr":gaussianProcessRegression,
"hr":huberRegression,
"tsr":theilSenRegression,
"par":passiveAggressiveRegression,
"ard":ardRegression,
"bays_ridge":bayesianRidgeRegression,
"lasso_lar":lassoLeastAngleRegression,
"lar":leastAngleRegression}
</p>

<p>Regression metrics:
['rmse','mse', 'r2','explained_variance','mean_absolute_error','median_absolute_error']
</p>
<p>Classification Models:

	{'abc':adaptiveBoostingClassifier,
	'dtc':decisionTreeClassifier,
	'gbc':gradientBoostingClassifier,
	'gpc':guassianProcessClassifier,
	'knnc':knnClassifier,
	'logr':logisticRegressionClassifier,
	'mlpc':multilayerPerceptronClassifier,
	'nbc':naiveBayesClassifier,
	'rfc':randomForestClassifier,
	'sgd':stochasticGradientDescentClassifier,
	'svc':supportVectorClassifier}
			
</p>
<p>Classification Metrics:
["auc","prec","recall","f1","accuracy", "kappa","log_loss"]
</p>
</li>

<li><code>FeatureSelectionStream</code>, meant to flow through several predictive models and algorithms to determine which subset of features is most predictive or representative of your dataset, these include: RandomForestFeatureImportance, LassoFeatureImportance, MixedSelection, and a technique to ensemble each named TOPSISFeatureRanking. You must specify whether your wish to ensemble and with what technique (denoted <code>ensemble=True). This is not currently supported, however will be built on top the <em>sklearn.feature_selection</em>.</code> 
</li>
</ul>

<hr>

<h2>Some Examples</h2>
<strong>Simple data set</strong>
<code>
	
	X = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(200)]))
	
	y = pd.DataFrame(np.array([np.random.exponential() for i in range(200)]))
</code>

<strong>Supported stream operators</strong>: scale, normalize, boxcox, binarize, pca, kmeans, brbm (Bernoulli Restricted Boltzman Machine).

<code> 

	Xnew = TransformationStream(X).flow(
		["scale","normalize","pca", "binarize", "boxcox", "kmeans", "brbm"],
	 	params={"pca__percent_variance":0.75, 
				"kmeans__n_clusters":2, 
				"binarize__threshold":0.5, 
				"brbm__n_components":X.shape[1], 
				"brbm__learning_rate":0.0001},
				verbose=True)
</code>

<code>
<h2>Regression</h2>

	performances = ModelSelectionStream(Xnew,y).flow(
		["svr", "lr", "knnr","lasso","abr","mlp","enet"],

	    params={'svr__C':[1,0.1,0.01,0.001],

				'svr__gamma':[0, 0.01, 0.001, 0.0001],
				
				'svr__kernel':['poly', 'rbf'],
				
				'svr__epsilon':[0,0.1,0.01,0.001],
				
				'svr__degree':[1,2,3,4,5,6,7],
				
				'lr__fit_intercept':[False, True],
				
				'knnr__n_neighbors':[3, 5,7, 9, 11, 13],
				
				'lasso__alpha':[0, 0.1, 0.01,1,10.0,20.0],
				
				'ridge__alpha':[0, 0.1, 0.01,1,10.0,20.0],
				
				'enet__alpha':[0, 0.1, 0.01,1,10,20],
				
				'enet__l1_ratio':[.25,.5,.75],
				
				'abr__n_estimators':[10,20,50],
				
				'abr__learning_rate':[0.1,1,10, 100],
				
				'rfr__criterion':['mse', 'mae'],
				
				'rfr__n_estimators':[10,100,1000]}, 
				
		metrics=['r2','rmse', 'mse',
		'explained_variance','mean_absolute_error',
		'median_absolute_error'],
		verbose=True,
		regressors=True,
		cut=2)
</code>
<code>
# Classification
	performances = ModelSelectionStream(X2,y2).flow(
		["abc"], 
		
		params={'abc__n_estimators':[10,100,1000],
		'abc__learning_rate':[0.001,0.01,0.1,1,10,100]},
		
		metrics=["auc",
			   	"prec",
				"recall",
				"f1",
				"accuracy",
				"kappa",
				"log_loss"],
		verbose=True,
		modelSelection=True,
		regressors=False
		)

</code>


