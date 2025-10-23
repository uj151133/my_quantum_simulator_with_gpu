
// Generated from OpenQASM3.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  OpenQASM3Parser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, T__9 = 10, T__10 = 11, GPHASE = 12, ID = 13, P = 14, 
    X = 15, Y = 16, Z = 17, SX = 18, H = 19, S = 20, SDG = 21, T = 22, TDG = 23, 
    RX = 24, RY = 25, RZ = 26, CX = 27, CY = 28, CZ = 29, CP = 30, CRX = 31, 
    CRY = 32, CRZ = 33, CH = 34, CU = 35, SWAP = 36, CCX = 37, CSWAP = 38, 
    U1 = 39, U2 = 40, U3 = 41, RESET = 42, BARRIER = 43, MEASURE = 44, NUMBER = 45, 
    IDSTR = 46, STRING = 47, LBRACKET = 48, RBRACKET = 49, LPAREN = 50, 
    RPAREN = 51, COMMA = 52, SEMICOLON = 53, WS = 54
  };

  enum {
    RuleProgram = 0, RuleIncludeStmt = 1, RuleStatement = 2, RuleQregDecl = 3, 
    RuleCregDecl = 4, RuleGateStmt = 5, RuleParamList = 6, RuleExpr = 7, 
    RuleGateName = 8, RuleGateArgs = 9, RuleQubit = 10, RuleMeasureStmt = 11, 
    RuleBarrierStmt = 12, RuleQubitList = 13
  };

  explicit OpenQASM3Parser(antlr4::TokenStream *input);

  OpenQASM3Parser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~OpenQASM3Parser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class ProgramContext;
  class IncludeStmtContext;
  class StatementContext;
  class QregDeclContext;
  class CregDeclContext;
  class GateStmtContext;
  class ParamListContext;
  class ExprContext;
  class GateNameContext;
  class GateArgsContext;
  class QubitContext;
  class MeasureStmtContext;
  class BarrierStmtContext;
  class QubitListContext; 

  class  ProgramContext : public antlr4::ParserRuleContext {
  public:
    ProgramContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> NUMBER();
    antlr4::tree::TerminalNode* NUMBER(size_t i);
    antlr4::tree::TerminalNode *SEMICOLON();
    std::vector<IncludeStmtContext *> includeStmt();
    IncludeStmtContext* includeStmt(size_t i);
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProgramContext* program();

  class  IncludeStmtContext : public antlr4::ParserRuleContext {
  public:
    IncludeStmtContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STRING();
    antlr4::tree::TerminalNode *SEMICOLON();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IncludeStmtContext* includeStmt();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QregDeclContext *qregDecl();
    CregDeclContext *cregDecl();
    GateStmtContext *gateStmt();
    MeasureStmtContext *measureStmt();
    BarrierStmtContext *barrierStmt();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StatementContext* statement();

  class  QregDeclContext : public antlr4::ParserRuleContext {
  public:
    QregDeclContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDSTR();
    antlr4::tree::TerminalNode *LBRACKET();
    antlr4::tree::TerminalNode *NUMBER();
    antlr4::tree::TerminalNode *RBRACKET();
    antlr4::tree::TerminalNode *SEMICOLON();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QregDeclContext* qregDecl();

  class  CregDeclContext : public antlr4::ParserRuleContext {
  public:
    CregDeclContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDSTR();
    antlr4::tree::TerminalNode *LBRACKET();
    antlr4::tree::TerminalNode *NUMBER();
    antlr4::tree::TerminalNode *RBRACKET();
    antlr4::tree::TerminalNode *SEMICOLON();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CregDeclContext* cregDecl();

  class  GateStmtContext : public antlr4::ParserRuleContext {
  public:
    GateStmtContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    GateNameContext *gateName();
    GateArgsContext *gateArgs();
    antlr4::tree::TerminalNode *SEMICOLON();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ParamListContext *paramList();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  GateStmtContext* gateStmt();

  class  ParamListContext : public antlr4::ParserRuleContext {
  public:
    ParamListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ParamListContext* paramList();

  class  ExprContext : public antlr4::ParserRuleContext {
  public:
    ExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *NUMBER();
    antlr4::tree::TerminalNode *IDSTR();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *RPAREN();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExprContext* expr();
  ExprContext* expr(int precedence);
  class  GateNameContext : public antlr4::ParserRuleContext {
  public:
    GateNameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *X();
    antlr4::tree::TerminalNode *Y();
    antlr4::tree::TerminalNode *Z();
    antlr4::tree::TerminalNode *H();
    antlr4::tree::TerminalNode *S();
    antlr4::tree::TerminalNode *SDG();
    antlr4::tree::TerminalNode *T();
    antlr4::tree::TerminalNode *TDG();
    antlr4::tree::TerminalNode *RX();
    antlr4::tree::TerminalNode *RY();
    antlr4::tree::TerminalNode *RZ();
    antlr4::tree::TerminalNode *CX();
    antlr4::tree::TerminalNode *CY();
    antlr4::tree::TerminalNode *CZ();
    antlr4::tree::TerminalNode *CP();
    antlr4::tree::TerminalNode *SWAP();
    antlr4::tree::TerminalNode *CCX();
    antlr4::tree::TerminalNode *CSWAP();
    antlr4::tree::TerminalNode *U1();
    antlr4::tree::TerminalNode *U2();
    antlr4::tree::TerminalNode *U3();
    antlr4::tree::TerminalNode *SX();
    antlr4::tree::TerminalNode *CH();
    antlr4::tree::TerminalNode *CRX();
    antlr4::tree::TerminalNode *CRY();
    antlr4::tree::TerminalNode *CRZ();
    antlr4::tree::TerminalNode *CU();
    antlr4::tree::TerminalNode *P();
    antlr4::tree::TerminalNode *ID();
    antlr4::tree::TerminalNode *GPHASE();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  GateNameContext* gateName();

  class  GateArgsContext : public antlr4::ParserRuleContext {
  public:
    GateArgsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<QubitContext *> qubit();
    QubitContext* qubit(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  GateArgsContext* gateArgs();

  class  QubitContext : public antlr4::ParserRuleContext {
  public:
    QubitContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDSTR();
    antlr4::tree::TerminalNode *LBRACKET();
    antlr4::tree::TerminalNode *NUMBER();
    antlr4::tree::TerminalNode *RBRACKET();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QubitContext* qubit();

  class  MeasureStmtContext : public antlr4::ParserRuleContext {
  public:
    MeasureStmtContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MEASURE();
    QubitContext *qubit();
    antlr4::tree::TerminalNode *IDSTR();
    antlr4::tree::TerminalNode *LBRACKET();
    antlr4::tree::TerminalNode *NUMBER();
    antlr4::tree::TerminalNode *RBRACKET();
    antlr4::tree::TerminalNode *SEMICOLON();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  MeasureStmtContext* measureStmt();

  class  BarrierStmtContext : public antlr4::ParserRuleContext {
  public:
    BarrierStmtContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *BARRIER();
    QubitListContext *qubitList();
    antlr4::tree::TerminalNode *SEMICOLON();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BarrierStmtContext* barrierStmt();

  class  QubitListContext : public antlr4::ParserRuleContext {
  public:
    QubitListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<QubitContext *> qubit();
    QubitContext* qubit(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QubitListContext* qubitList();


  bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;

  bool exprSempred(ExprContext *_localctx, size_t predicateIndex);

  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

