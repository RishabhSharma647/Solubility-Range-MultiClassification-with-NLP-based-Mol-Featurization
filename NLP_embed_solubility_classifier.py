""" MLP Network for Multi-Class Classification of SMILES Into Solubility Ranges 
Using NLP (skip gram)Based Embedding/Feauturizations Using Mol2Vec in TF 2.0 """

import mol2vec
import numpy as np
import pandas as pd
import rdkit
import tensorflow as tf

from gensim.models import word2vec
from MLP_Solubility_Range_Classifier import MLP_Classifier 
from mol2vec.features import DfVec
from mol2vec.features import mol2alt_sentence
from mol2vec.features import mol2sentence
from mol2vec.features import MolSentence
from mol2vec.features import sentences2vec
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dropout

# Import file containing SMILES with solubility values in ascending order
# (Subset of AqSolDB) containing SMILES valid for Mol2vec 
sol_sorted_data = pd.read_csv('Sorted_processed_sol_data.csv')

# Import pre-trained mol2vec model
mol2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')

# Define function to convert SMILES strings to 300d embeddings
def smiles2vector_duplicates_average(smiles_string):
    """ Convert SMILES to 300d embedding
    
    Args:
        smiles_string (string): single SMILES string
    
    Returns:
        embedding (numpy.ndarray): 300d mol vector array
    
    """
    sentence = mol2alt_sentence(Chem.MolFromSmiles(smiles_string), radius = 1)

    vec_node = 0
    for i in range(len(sentence)):
        vec = mol2vec_model.wv[sentence[i]]
        vec_node += vec

    return vec_node / len(sentence)


# Create training data
# Vectorize SMILES using Mol2vec
all_mol2vecs = [list(smiles2vector_duplicates_average(smile)) for smile in sol_sorted_data['SMILES']]


# Create input and output data for NLP (skip gram) based MLP model for embedding Mol2vec features
# using sequential SMILES with ascending solubiluty values
# Inputs: Standard mol2vec vector; Outputs: Mean mol2vec vectors in window of --> last two and following two SMILES
# in sequence of SMILES with ascending solubility values 
in_vectors, out_vectors = [], []
window = 2
for i in range(window, len(all_mol2vecs) - window):
    
    in_vectors.append(all_mol2vecs[i])
    
    first = np.array(all_mol2vecs[i - window])
    second = np.array(all_mol2vecs[i - 1])
    third = np.array(all_mol2vecs[i + 1])
    fourth = np.array(all_mol2vecs[i + window])
    
    add = list(np.mean([first, second, third, fourth], axis = 0))
    out_vectors.append(add)
    

# Create input and output dataframes
columns = ['mol2vec_' + str(i) for i in range(1, 301)]
df_embed_inputs = pd.DataFrame(in_vectors, columns = columns)
df_embed_outputs = pd.DataFrame(out_vectors, columns = columns)

# Define input/output array 
X = df_embed_inputs.iloc[:, 0:300].values
y = df_embed_outputs.iloc[:, 0:300].values 

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Scale data 
scaler = StandardScaler()
X_trans = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# Define sequential MLP model to map input mol2vec vector to output mean of mol2vec vectors
# of last two and following two SMILES in sequence of SMILES with ascending solubility values 
model_embedder = tf.keras.Sequential()
model_embedder.add(layers.Dense(250, input_dim = 300, activation='tanh')) 
model_embedder.add(Dropout(0.3))
model_embedder.add(layers.Dense(290, activation='tanh'))
model_embedder.add(Dropout(0.3))
model_embedder.add(layers.Dense(300))

# Compile model using MAE loss and adam optimizer
model_embedder.compile(loss='mae', optimizer='adam')

# Define batch size and number of epochs
batch_size = 50
num_epochs = 50

# Fit model
model_embedder.fit(X_trans, y_train, 
          epochs = num_epochs,
          batch_size = batch_size,
          validation_split = .2, shuffle = True)

# Test results (use last model as higher validation accuracies stabalize towards later epochs)
results = model_embedder.evaluate(X_test, y_test)
print(f"MAE (test): {results:.3f}")

# Define function to retreive second hidden layer's representations of input vectors, to be used as inputs
# for multi-class solubility range classification
get_hidden_outputs = K.function(inputs = model_embedder.layers[0].input, outputs = model_embedder.layers[2].output)


""" Make new MLP model for solubility range classificaton using hidden layer vector embeddings of model """

# Import processed solubility data file (containing SMILES valid for Mol2vec vectorization) 
# (Subset of AqSolDB)
sol_data = pd.read_csv('processed_sol_data.csv')

# Classify solubility values into range labels
all_sols = list(sol_data['Solubility_logS'])

# Define function to map solubility values to range classes
# (Defined in order to enforce fair class balance along with range values)
def map_solubility_to_range(sol):
    if min(all_sols) <= sol <= -5:
        return 0
    elif -5 < sol <= -3:
        return 1
    elif -3 < sol <= -1.3:
        return 2
    elif -1.3 < sol <= max(all_sols):
        return 3


# Sort solubility values into solubility range labels
range_labels = [map_solubility_to_range(sol) for sol in all_sols]
# Add column for range labels to dataframe
sol_data['range_labels'] = range_labels

# Add one hot encodings 
sol_data['one_hot_1'] = [1.0 if sol_data['range_labels'][i] == 0 else 0.0 for i in range(sol_data.shape[0])]
sol_data['one_hot_2'] = [1.0 if sol_data['range_labels'][i] == 1 else 0.0 for i in range(sol_data.shape[0])]
sol_data['one_hot_3'] = [1.0 if sol_data['range_labels'][i] == 2 else 0.0 for i in range(sol_data.shape[0])]
sol_data['one_hot_4'] = [1.0 if sol_data['range_labels'][i] == 3 else 0.0 for i in range(sol_data.shape[0])]

# Creat input and output data arrays
all_rows_df = [list(smiles2vector_duplicates_average(smile)) for smile in sol_data['SMILES']]
df_inputs = pd.DataFrame(all_rows_df, columns = columns)

X_sol_class = df_inputs.iloc[:, 0:300].values
X_sol_class = scaler.transform(X_sol_class) 

# Create hidden layer representations to be used as inputs for solubility range classification model
X_featurized = np.array([get_hidden_outputs(np.array([X_sol_class[i],]))[0] for i in range(len(X_sol_class))])
y = sol_data.iloc[:, 4:8].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_featurized, y, test_size=0.15, random_state=42)

# Scale data
scaler_two = StandardScaler()
X_trans = scaler_two.fit_transform(X_train)
X_test = scaler_two.transform(X_test) 

# Define hyper parameters (Identical to those used in solubility range classifier using standard Mol2vec vector inputs)
hidden_layers = {'h1': [200, 'tanh'], 'h2': [100, 'tanh'], 
                 'h3': [50, 'tanh'], 'output': [4, 'softmax']}
dropout = 0.3
loss = 'categorical_crossentropy'
optimizer = 'adam'
metric = 'accuracy'
filepath="nlp_model_classifier_weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
batch_size = 100
num_epochs = 200
validation_split = .2

# Instantiate and create model
mlp_sol_classifier = MLP_Classifier(hidden_layers, dropout)

# Compile model
mlp_sol_classifier.compile_model(loss, optimizer, metric)

# Fit model
mlp_sol_classifier.fit_model(callbacks_list, batch_size, num_epochs, validation_split, X_trans, y_train)

