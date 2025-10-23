
// Generated from OpenQASM3.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "OpenQASM3Visitor.h"


/**
 * This class provides an empty implementation of OpenQASM3Visitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  OpenQASM3BaseVisitor : public OpenQASM3Visitor {
public:

  virtual std::any visitProgram(OpenQASM3Parser::ProgramContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIncludeStmt(OpenQASM3Parser::IncludeStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStatement(OpenQASM3Parser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitQregDecl(OpenQASM3Parser::QregDeclContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCregDecl(OpenQASM3Parser::CregDeclContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGateStmt(OpenQASM3Parser::GateStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParamList(OpenQASM3Parser::ParamListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpr(OpenQASM3Parser::ExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGateName(OpenQASM3Parser::GateNameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGateArgs(OpenQASM3Parser::GateArgsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitQubit(OpenQASM3Parser::QubitContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMeasureStmt(OpenQASM3Parser::MeasureStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBarrierStmt(OpenQASM3Parser::BarrierStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitQubitList(OpenQASM3Parser::QubitListContext *ctx) override {
    return visitChildren(ctx);
  }


};

