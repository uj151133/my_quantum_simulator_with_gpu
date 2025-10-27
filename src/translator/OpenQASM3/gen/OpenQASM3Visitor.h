
// Generated from OpenQASM3.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "OpenQASM3Parser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by OpenQASM3Parser.
 */
class  OpenQASM3Visitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by OpenQASM3Parser.
   */
    virtual std::any visitProgram(OpenQASM3Parser::ProgramContext *context) = 0;

    virtual std::any visitIncludeStmt(OpenQASM3Parser::IncludeStmtContext *context) = 0;

    virtual std::any visitStatement(OpenQASM3Parser::StatementContext *context) = 0;

    virtual std::any visitQregDecl(OpenQASM3Parser::QregDeclContext *context) = 0;

    virtual std::any visitCregDecl(OpenQASM3Parser::CregDeclContext *context) = 0;

    virtual std::any visitGateStmt(OpenQASM3Parser::GateStmtContext *context) = 0;

    virtual std::any visitParamList(OpenQASM3Parser::ParamListContext *context) = 0;

    virtual std::any visitExpr(OpenQASM3Parser::ExprContext *context) = 0;

    virtual std::any visitGateName(OpenQASM3Parser::GateNameContext *context) = 0;

    virtual std::any visitGateArgs(OpenQASM3Parser::GateArgsContext *context) = 0;

    virtual std::any visitQubit(OpenQASM3Parser::QubitContext *context) = 0;

    virtual std::any visitMeasureStmt(OpenQASM3Parser::MeasureStmtContext *context) = 0;

    virtual std::any visitBarrierStmt(OpenQASM3Parser::BarrierStmtContext *context) = 0;

    virtual std::any visitQubitList(OpenQASM3Parser::QubitListContext *context) = 0;


};

