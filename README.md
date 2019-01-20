<h1>Streamlined Machine Learning</h1>
<hr>
<strong>Streamlined Machine Learning</strong> is a high level machine learning workflow wrapper built around sklearn and scipy.stats. There exist three primary functions: 1. Transformation Preprocessing 2. Model Selection 3. Feature Selection. <em>streamml</em> contains a set of robust functions built on top of the <em>sklearn</em> framework and aims at streamlining all of the processes in the context of a flow.
<hr>
<h2> Installation </h2>
	<code> pip install -U streamml2 </code>
	
<hr>
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

<li>
<code>FeatureSelectionStream</code>, meant to flow through several predictive models and algorithms to determine which subset of features is most predictive or representative of your dataset, these include: RandomForestFeatureImportance, LassoFeatureImportance, MixedSelection, and a technique to ensemble each named TOPSISFeatureRanking. You must specify whether your wish to ensemble and with what technique (denoted <code>ensemble=True). This is not currently supported, however will be built on top the <em>sklearn.feature_selection</em>.</code> 
</li>
</ul>

<hr>

<strong>Supported stream operators</strong>: scale, normalize, boxcox, binarize, pca, kmeans, brbm (Bernoulli Restricted Boltzman Machine).</strong>

<h2> Transformation </h2>
<code> 

	import pandas as pd

	from streamml2.streamml2.streamline.transformation.flow.TransformationStream import TransformationStream

	from sklearn.datasets import fetch_20newsgroups

	categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']

	newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)

	X2 = TransformationStream(newsgroups_train.data,corpus=True, method='tfidf').flow(["pca","normalize","kmeans"],

					  params={"pca__percent_variance":0.95,

						  "kmeans__n_clusters":len(categories)})

	print(X2)

	from sklearn.datasets import load_iris

	iris=load_iris()

	X=pd.DataFrame(iris['data'], columns=iris['feature_names'])

	y=pd.DataFrame(iris['target'], columns=['target'])

	X2 = TransformationStream(X).flow(["pca","scale","normalize","kmeans"],

					  params={"pca__n_components":2,

						  "kmeans__n_clusters":len(set(y['target']))})

	print(X2)

</code>

<h2>Regression</h2>
<code>
	
	from streamml2.streamml2.streamline.model_selection.flow.ModelSelectionStream import ModelSelectionStream

	from sklearn.svm import SVR

	from sklearn.ensemble import RandomForestRegressor

	from sklearn.linear_model import LinearRegression

	import pandas as pd

	from sklearn.datasets import load_boston

	boston=load_boston()

	X=pd.DataFrame(boston['data'], columns=boston['feature_names'])

	y=pd.DataFrame(boston['target'],columns=["target"])

	regression_options={"lr" : 0,
			   "svr" : 0,
			   "rfr":0,
			   "abr":0,
			   "knnr":0,
			   "ridge":0,
			   "lasso":0,
			   "enet":0,
			   "mlpr":0,
			   "br":0,
			   "dtr":0,
			   "gbr":0,
			   "gpr":0,
			   "hr":0,
			   "tsr":0,
			   "par":0,
			   "ard":0,
			   "bays_ridge":0,
			   "lasso_lar":0,
			   "lar":0}
	results_dict = ModelSelectionStream(X,y).flow(list(regression_options.keys()),
									    params={},
									    metrics=[],
									    test_size=0.5,
									    nfolds=10,
									    nrepeats=10,
									    verbose=False, 
									    regressors=True,
									    stratified=True, 
									    cut=y['target'].mean(),
									    modelSelection=True,
									    n_jobs=3)

	print("Best Models ... ")
	print(results_dict["models"])
	print("Final Errors ... ")
	print(pd.DataFrame(results_dict["final_errors"]))
	print("Metric Table ...")
	print(pd.DataFrame(results_dict["avg_kfold"]))
	print("Significance By Metric ...")
	for k in results_dict["significance"].keys():
	    print(k)
	    print(results_dict["significance"][k])
    
</code>

<h2>Feature Selection</h2>
<code>
	
	from sklearn.datasets import load_iris
	iris=load_iris()
	X=pd.DataFrame(iris['data'], columns=iris['feature_names'])
	y=pd.DataFrame(iris['target'], columns=['target'])

	return_dict = FeatureSelectionStream(X,y).flow(["rfc", "abc", "svc"],
							params={},
							verbose=True,
							regressors=False,
							ensemble=True,
							featurePercentage=0.5,
							n_jobs=3)

	print("Feature data ...")
	print(pd.DataFrame(return_dict['feature_importances']))
	print("Features rankings decision maker...")
	print(return_dict['ensemble_results'])
	print("Reduced data ...")
	print(X[return_dict['kept_features']].head())

	from sklearn.datasets import load_boston
	boston=load_boston()
	X=pd.DataFrame(boston['data'], columns=boston['feature_names'])
	y=pd.DataFrame(boston['target'],columns=["target"])

	return_dict = FeatureSelectionStream(X,y).flow(["plsr", "mixed_selection", "rfr", "abr", "svr"],
							params={"mixed_selection__threshold_in":0.01,
								"mixed_selection__threshold_out":0.05,
								"mixed_selection__verbose":True},
							verbose=True,
							regressors=True,
							ensemble=True,
							featurePercentage=0.5,
							n_jobs=3)
						
</code>
