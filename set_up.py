from qiskit_ibm_runtime import QiskitRuntimeService
from dotenv import load_dotenv
import os


load_dotenv()
token = os.getenv('IBM_QUANTUM_TOKEN')
service = QiskitRuntimeService(channel='ibm_quantum', token=token)
