# Solubility Range Multi-class Classification with NLP based Featurization

Aqueous Solubility Range Multi-class classification models in TensorFlow 2.0, using standard Mol2Vec featurizer and a property-aware NLP (skip-gram) adapted/customised featurizer for SMILES molecules. 

### Dependencies

[Numpy](https://anaconda.org/conda-forge/numpy)

[Pandas](https://anaconda.org/anaconda/pandas)

[TensorFlow 2.0](https://www.tensorflow.org/install)

[RDKit](https://www.rdkit.org/docs/Install.html)

[Mol2vec](https://github.com/samoturk/mol2vec)

[Gensim](https://anaconda.org/anaconda/gensim)

[Scikit-learn](https://anaconda.org/anaconda/scikit-learn)

[300-dim mol2vec model](https://github.com/samoturk/mol2vec/blob/master/examples/models/model_300dim.pkl)

### Featurization

SMILES representation of input molecules are featurized using an unsupervised pre-trained NLP-based model, as introduced by Samo Turk et al in [Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00616). Analagous to Word2vec models that vectorize words such that semantically proximate words are clustered together in the vector space, this approach learns vector representations of molecular substructures that are proximate for chemically related substructures, which is suitable for featurizing molecules for supervised machine learning methods. 

