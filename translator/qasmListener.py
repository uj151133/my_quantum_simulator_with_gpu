# Generated from qasm.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .qasmParser import qasmParser
else:
    from qasmParser import qasmParser

# This class defines a complete listener for a parse tree produced by qasmParser.
class qasmListener(ParseTreeListener):

    # Enter a parse tree produced by qasmParser#mainprog.
    def enterMainprog(self, ctx:qasmParser.MainprogContext):
        pass

    # Exit a parse tree produced by qasmParser#mainprog.
    def exitMainprog(self, ctx:qasmParser.MainprogContext):
        pass


    # Enter a parse tree produced by qasmParser#header.
    def enterHeader(self, ctx:qasmParser.HeaderContext):
        pass

    # Exit a parse tree produced by qasmParser#header.
    def exitHeader(self, ctx:qasmParser.HeaderContext):
        pass


    # Enter a parse tree produced by qasmParser#qregDecl.
    def enterQregDecl(self, ctx:qasmParser.QregDeclContext):
        pass

    # Exit a parse tree produced by qasmParser#qregDecl.
    def exitQregDecl(self, ctx:qasmParser.QregDeclContext):
        pass


    # Enter a parse tree produced by qasmParser#cregDecl.
    def enterCregDecl(self, ctx:qasmParser.CregDeclContext):
        pass

    # Exit a parse tree produced by qasmParser#cregDecl.
    def exitCregDecl(self, ctx:qasmParser.CregDeclContext):
        pass


    # Enter a parse tree produced by qasmParser#iGate.
    def enterIGate(self, ctx:qasmParser.IGateContext):
        pass

    # Exit a parse tree produced by qasmParser#iGate.
    def exitIGate(self, ctx:qasmParser.IGateContext):
        pass


    # Enter a parse tree produced by qasmParser#xGate.
    def enterXGate(self, ctx:qasmParser.XGateContext):
        pass

    # Exit a parse tree produced by qasmParser#xGate.
    def exitXGate(self, ctx:qasmParser.XGateContext):
        pass


    # Enter a parse tree produced by qasmParser#zGate.
    def enterZGate(self, ctx:qasmParser.ZGateContext):
        pass

    # Exit a parse tree produced by qasmParser#zGate.
    def exitZGate(self, ctx:qasmParser.ZGateContext):
        pass


    # Enter a parse tree produced by qasmParser#sGate.
    def enterSGate(self, ctx:qasmParser.SGateContext):
        pass

    # Exit a parse tree produced by qasmParser#sGate.
    def exitSGate(self, ctx:qasmParser.SGateContext):
        pass


    # Enter a parse tree produced by qasmParser#sdgGate.
    def enterSdgGate(self, ctx:qasmParser.SdgGateContext):
        pass

    # Exit a parse tree produced by qasmParser#sdgGate.
    def exitSdgGate(self, ctx:qasmParser.SdgGateContext):
        pass


    # Enter a parse tree produced by qasmParser#hGate.
    def enterHGate(self, ctx:qasmParser.HGateContext):
        pass

    # Exit a parse tree produced by qasmParser#hGate.
    def exitHGate(self, ctx:qasmParser.HGateContext):
        pass


    # Enter a parse tree produced by qasmParser#tGate.
    def enterTGate(self, ctx:qasmParser.TGateContext):
        pass

    # Exit a parse tree produced by qasmParser#tGate.
    def exitTGate(self, ctx:qasmParser.TGateContext):
        pass


    # Enter a parse tree produced by qasmParser#tdgGate.
    def enterTdgGate(self, ctx:qasmParser.TdgGateContext):
        pass

    # Exit a parse tree produced by qasmParser#tdgGate.
    def exitTdgGate(self, ctx:qasmParser.TdgGateContext):
        pass


    # Enter a parse tree produced by qasmParser#swapGate.
    def enterSwapGate(self, ctx:qasmParser.SwapGateContext):
        pass

    # Exit a parse tree produced by qasmParser#swapGate.
    def exitSwapGate(self, ctx:qasmParser.SwapGateContext):
        pass


    # Enter a parse tree produced by qasmParser#cxGate.
    def enterCxGate(self, ctx:qasmParser.CxGateContext):
        pass

    # Exit a parse tree produced by qasmParser#cxGate.
    def exitCxGate(self, ctx:qasmParser.CxGateContext):
        pass


    # Enter a parse tree produced by qasmParser#czGate.
    def enterCzGate(self, ctx:qasmParser.CzGateContext):
        pass

    # Exit a parse tree produced by qasmParser#czGate.
    def exitCzGate(self, ctx:qasmParser.CzGateContext):
        pass


    # Enter a parse tree produced by qasmParser#rxGate.
    def enterRxGate(self, ctx:qasmParser.RxGateContext):
        pass

    # Exit a parse tree produced by qasmParser#rxGate.
    def exitRxGate(self, ctx:qasmParser.RxGateContext):
        pass


    # Enter a parse tree produced by qasmParser#ryGate.
    def enterRyGate(self, ctx:qasmParser.RyGateContext):
        pass

    # Exit a parse tree produced by qasmParser#ryGate.
    def exitRyGate(self, ctx:qasmParser.RyGateContext):
        pass


    # Enter a parse tree produced by qasmParser#rzGate.
    def enterRzGate(self, ctx:qasmParser.RzGateContext):
        pass

    # Exit a parse tree produced by qasmParser#rzGate.
    def exitRzGate(self, ctx:qasmParser.RzGateContext):
        pass


    # Enter a parse tree produced by qasmParser#u3Gate.
    def enterU3Gate(self, ctx:qasmParser.U3GateContext):
        pass

    # Exit a parse tree produced by qasmParser#u3Gate.
    def exitU3Gate(self, ctx:qasmParser.U3GateContext):
        pass


    # Enter a parse tree produced by qasmParser#measureStmt.
    def enterMeasureStmt(self, ctx:qasmParser.MeasureStmtContext):
        pass

    # Exit a parse tree produced by qasmParser#measureStmt.
    def exitMeasureStmt(self, ctx:qasmParser.MeasureStmtContext):
        pass


    # Enter a parse tree produced by qasmParser#expr.
    def enterExpr(self, ctx:qasmParser.ExprContext):
        pass

    # Exit a parse tree produced by qasmParser#expr.
    def exitExpr(self, ctx:qasmParser.ExprContext):
        pass


    # Enter a parse tree produced by qasmParser#exprMul.
    def enterExprMul(self, ctx:qasmParser.ExprMulContext):
        pass

    # Exit a parse tree produced by qasmParser#exprMul.
    def exitExprMul(self, ctx:qasmParser.ExprMulContext):
        pass


    # Enter a parse tree produced by qasmParser#sign.
    def enterSign(self, ctx:qasmParser.SignContext):
        pass

    # Exit a parse tree produced by qasmParser#sign.
    def exitSign(self, ctx:qasmParser.SignContext):
        pass



del qasmParser