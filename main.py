import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
import os

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('Quantum Circuit and Bloch Sphere')
        self.setGeometry(100, 100, 1200, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.circuit_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.circuit_canvas)

        self.bloch_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.bloch_canvas)

        self.draw_quantum_circuit()
        self.draw_bloch_sphere()

    def draw_quantum_circuit(self):
        with open('quantum_circuit.qasm', 'r') as f:
            qasm_code = f.read()
        qc = QuantumCircuit.from_qasm_str(qasm_code)

        qc.draw('mpl', ax=self.circuit_canvas.axes)
        self.circuit_canvas.draw()

    def draw_bloch_sphere(self):
        with open('quantum_circuit.qasm', 'r') as f:
            qasm_code = f.read()
        qc = QuantumCircuit.from_qasm_str(qasm_code)

        simulator = Aer.get_backend('statevector_simulator')
        qc_transpiled = transpile(qc, simulator)
        result = execute(qc_transpiled, backend=simulator).result()
        statevector = result.get_statevector()

        # ブロッホ球の描画を新しいFigureにプロットし、画像をキャンバスにコピー
        fig, ax = plt.subplots()
        plot_bloch_multivector(statevector)
        fig.savefig('bloch_sphere.png')
        plt.close(fig)

        # 画像を読み込んでキャンバスに表示
        self.bloch_canvas.axes.clear()
        img = plt.imread('bloch_sphere.png')
        self.bloch_canvas.axes.imshow(img)
        self.bloch_canvas.axes.axis('off')
        self.bloch_canvas.draw()

        # 一時ファイルを削除
        os.remove('bloch_sphere.png')

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
