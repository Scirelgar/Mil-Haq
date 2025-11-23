#Author : Xavier Bergeron ,  Lia Suci Waliani , Gabriel Lemay, Emeryck ALLAIN , Naomi Catwell
#This code was created during the Mil'HaQ , the goal was to predict the price of Swaption using a Quantum computing  	
import os
import multiprocessing
import numpy as np
import pandas as pd

import os
import multiprocessing
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


num_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)

EXCEL_PATH = "output_data.xlsx"
SAMPLE_N = None
TEST_SIZE = 0.3
VAL_SIZE = 0.1
RANDOM_STATE = 42
CHUNK_SIZE = 1000
USE_QRC = True
N_QUBITS = 3
N_Q_LAYERS = 2

rng_global = np.random.default_rng(RANDOM_STATE)
quantum_weights = rng_global.normal(loc=0.0, scale=0.5, size=(N_Q_LAYERS, N_QUBITS, 3))


def load_output_data(path=EXCEL_PATH, n_samples=None, random_state=RANDOM_STATE):
	df = pd.read_excel(path)
	df["Date"] = pd.to_datetime(df["Date"])
	df = df.sort_values("Date").reset_index(drop=True)
	if n_samples is not None and n_samples < len(df):
		df = df.sample(n=n_samples, random_state=random_state)
		df = df.sort_values("Date").reset_index(drop=True)
	return df


def build_features(df: pd.DataFrame):
	df = df.copy()
	df["time_index"] = (df["Date"] - df["Date"].min()).dt.days.astype(float)
	X = df[["Tenor (T)", "Maturity (τ)", "time_index"]].values.astype(float)
	y = df["Price (Y)"].values.astype(float)
	return X, y


def build_quantum_circuit(x, weights):
	qc = QuantumCircuit(N_QUBITS)
	scale = 0.1
	for i in range(N_QUBITS):
		angle = scale * float(x[i])
		qc.ry(angle, i)
	for layer in range(N_Q_LAYERS):
		for i in range(N_QUBITS):
			w0, w1, w2 = weights[layer, i]
			qc.rz(float(w0), i)
			qc.ry(float(w1), i)
			qc.rx(float(w2), i)
		for i in range(N_QUBITS - 1):
			qc.cx(i, i + 1)
	return qc


def z_expectations_from_statevector(statevector):
	statevector = np.asarray(statevector, dtype=complex)
	n_qubits = int(np.log2(statevector.size))
	probs = np.abs(statevector) ** 2
	expvals = np.zeros(n_qubits, dtype=float)
	for basis_index, p in enumerate(probs):
		for qubit in range(n_qubits):
			bit = (basis_index >> qubit) & 1
			expvals[qubit] += p * (1.0 if bit == 0 else -1.0)
	return expvals


def run_quantum_feature_map(x, weights):
	qc = build_quantum_circuit(x, weights)
	state = Statevector.from_instruction(qc)
	z_exps = z_expectations_from_statevector(state.data)
	return z_exps


def quantum_reservoir_features_qiskit(X, weights=quantum_weights):
	X = np.asarray(X, dtype=float)
	n_samples, d_in = X.shape
	if d_in != N_QUBITS:
		raise ValueError(f"Expected X to have {N_QUBITS} features for quantum encoding, but got {d_in}.")
	quantum_feats = np.zeros((n_samples, N_QUBITS), dtype=float)
	for i in range(n_samples):
		quantum_feats[i, :] = run_quantum_feature_map(X[i], weights)
	X_qrc = np.hstack([X, quantum_feats])
	return X_qrc


def prepare_train_val_test(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
	X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state)
	relative_val_size = val_size / (1.0 - test_size)
	X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=relative_val_size, shuffle=True, random_state=random_state + 1)
	x_scaler = StandardScaler()
	X_train_scaled = x_scaler.fit_transform(X_train)
	X_val_scaled = x_scaler.transform(X_val)
	X_test_scaled = x_scaler.transform(X_test)
	return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, x_scaler


def build_fast_regressor():
	model = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror", n_jobs=-1, tree_method="hist", random_state=RANDOM_STATE, eval_metric="rmse")
	return model


def train_with_progress(model, X_train, y_train, chunk_size=CHUNK_SIZE):
	n_samples = X_train.shape[0]
	for start in range(0, n_samples, chunk_size):
		end = min(start + chunk_size, n_samples)
	model.fit(X_train, y_train)
	return model


def main():
	df = load_output_data(EXCEL_PATH, n_samples=SAMPLE_N)
	X, y = build_features(df)
	if USE_QRC:
		X = quantum_reservoir_features_qiskit(X)
	X_train, X_val, X_test, y_train, y_val, y_test, x_scaler = prepare_train_val_test(X, y)
	model = build_fast_regressor()
	model = train_with_progress(model, X_train, y_train)
	y_pred = model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, y_pred)
	accuracy_pct = r2 * 100.0
	n_show = min(10, len(y_pred))
	for i in range(n_show):
		pass


if __name__ == "__main__":
	main()
	for basis_index, p in enumerate(probs):

		for qubit in range(n_qubits):

			bit = (basis_index >> qubit) & 1
