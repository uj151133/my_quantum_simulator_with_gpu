import os, re, sys
from antlr4 import *
from qasmLexer import qasmLexer
from qasmParser import qasmParser
from qasmListener import qasmListener

def qasm_filename_to_cpp_func(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')
    func = parts[0] + ''.join(part.capitalize() for part in parts[1:])
    return func

def generate_hpp(funcname, hpp_path):
    guard = f"{funcname.upper()}_HPP"
    content = f"""#ifndef {guard}
#define {guard}

#include <gsl/gsl_sf_bessel.h>
#include "../../../models/state.hpp"
#include "../../../common/mathUtils.hpp"
#include "../../../models/circuit.hpp"

using namespace std;

void {funcname}();

#endif
"""
    with open(hpp_path, "w") as f:
        f.write(content)

class QasmCustomListener(qasmListener):
    def __init__(self):
        self.lines = []

    def enterQregDecl(self, ctx):
        name = ctx.ID().getText()
        size = ctx.INT().getText()
        self.lines.append(f"QuantumCircuit {name}({size});")

    def enterCregDecl(self, ctx):
        name = ctx.ID().getText()
        size = ctx.INT().getText()
        self.lines.append(f"vector<int> {name}({size});")

    def enterIGate(self, ctx):
        name = ctx.ID().getText()
        index = ctx.INT().getText()
        self.lines.append(f"{name}.addI({index});")
    
    def enterXGate(self, ctx):
        name = ctx.ID().getText()
        index = ctx.INT().getText()
        self.lines.append(f"{name}.addX({index});")
    
    def enterZGate(self, ctx):
        name = ctx.ID().getText()
        index = ctx.INT().getText()
        self.lines.append(f"{name}.addZ({index});")

    def enterSGate(self, ctx):
        name = ctx.ID().getText()
        index = ctx.INT().getText()
        self.lines.append(f"{name}.addS({index});")

    def enterSdgGate(self, ctx):
        name = ctx.ID().getText()
        index = ctx.INT().getText()
        self.lines.append(f"{name}.addSdg({index});")

    def enterHGate(self, ctx):
        name = ctx.ID().getText()
        index = ctx.INT().getText()
        self.lines.append(f"{name}.addH({index});")

    def enterTGate(self, ctx):
        name = ctx.ID().getText()
        index = ctx.INT().getText()
        self.lines.append(f"{name}.addT({index});")

    def enterTdgGate(self, ctx):
        name = ctx.ID().getText()
        index = ctx.INT().getText()
        self.lines.append(f"{name}.addTdg({index});")
    
    def enterSwapGate(self, ctx):
        name1 = ctx.ID(0).getText()
        index1 = ctx.INT(0).getText()
        index2 = ctx.INT(1).getText()
        self.lines.append(f"{name1}.addSWAP({index1}, {index2});")

    def enterCxGate(self, ctx):
        name1 = ctx.ID(0).getText()
        index1 = ctx.INT(0).getText()
        index2 = ctx.INT(1).getText()
        self.lines.append(f"{name1}.addCX({index1}, {index2});")

    def enterCzGate(self, ctx):
        name1 = ctx.ID(0).getText()
        index1 = ctx.INT(0).getText()
        index2 = ctx.INT(1).getText()
        self.lines.append(f"{name1}.addCZ({index1}, {index2});")

    def enterRzGate(self, ctx):
        qname = ctx.ID().getText()
        qidx = ctx.INT().getText()
        expr = ctx.expr().getText()
        expr = format_expr(expr)
        self.lines.append(f"{qname}.addRz({qidx}, {expr});")

    def enterU3Gate(self, ctx):
        qname = ctx.ID().getText()
        qidx = ctx.INT().getText()
        expr1 = ctx.expr(0).getText()
        expr2 = ctx.expr(1).getText()
        expr3 = ctx.expr(2).getText()
        expr1 = format_expr(expr1)
        expr2 = format_expr(expr2)
        expr3 = format_expr(expr3)
        self.lines.append(f"{qname}.addU3({qidx}, {expr1}, {expr2}, {expr3});")
    
    def enterMeasureStmt(self, ctx):
        c_name = ctx.ID(1).getText()
        c_idx = ctx.INT(1).getText()
        q_name = ctx.ID(0).getText()
        q_idx = ctx.INT(0).getText()
        self.lines.append(f"{c_name}[{c_idx}] = {q_name}.read({q_idx});")

def format_expr(expr):
    expr = expr.replace('pi', 'M_PI')
    expr = re.sub(r'\s*\*\s*', ' * ', expr)
    return expr

def main(qasm_filename):
    input_stream = FileStream(qasm_filename)
    lexer = qasmLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = qasmParser(stream)
    tree = parser.mainprog()

    listener = QasmCustomListener()
    walker = ParseTreeWalker()
    walker.walk(listener, tree)

    funcname = qasm_filename_to_cpp_func(qasm_filename)
    qasm_dir = os.path.dirname(os.path.abspath(qasm_filename))
    cpp_filename = os.path.join(qasm_dir, f"{funcname}.cpp")
    cpp_filename = os.path.abspath(cpp_filename)
    hpp_filename = os.path.join(qasm_dir, f"{funcname}.hpp")
    hpp_filename = os.path.abspath(hpp_filename)

    # 先にヘッダファイル生成
    generate_hpp(funcname, hpp_filename)

    with open(cpp_filename, "w") as f:
        f.write(f'#include "{funcname}.hpp"\n\n')
        f.write(f"void {funcname}() {{\n")
        for line in listener.lines:
            stripped = line.strip()
            if stripped == '' or stripped.startswith("//"):
                f.write("\n")
            else:
                f.write(f"    {line}\n")
        f.write(f"    return;\n")
        f.write("}\n")

    print(f"C++ファイルを生成しました: {cpp_filename}")
    print(f"ヘッダファイルを生成しました: {hpp_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <QASMファイル>")
        sys.exit(1)
    main(sys.argv[1])