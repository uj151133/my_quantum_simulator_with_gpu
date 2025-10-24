
// Generated from OpenQASM3.g4 by ANTLR 4.13.2


#include "OpenQASM3Visitor.h"

#include "OpenQASM3Parser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct OpenQASM3ParserStaticData final {
  OpenQASM3ParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  OpenQASM3ParserStaticData(const OpenQASM3ParserStaticData&) = delete;
  OpenQASM3ParserStaticData(OpenQASM3ParserStaticData&&) = delete;
  OpenQASM3ParserStaticData& operator=(const OpenQASM3ParserStaticData&) = delete;
  OpenQASM3ParserStaticData& operator=(OpenQASM3ParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag openqasm3ParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
std::unique_ptr<OpenQASM3ParserStaticData> openqasm3ParserStaticData = nullptr;

void openqasm3ParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (openqasm3ParserStaticData != nullptr) {
    return;
  }
#else
  assert(openqasm3ParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<OpenQASM3ParserStaticData>(
    std::vector<std::string>{
      "program", "includeStmt", "statement", "qregDecl", "cregDecl", "gateStmt", 
      "paramList", "expr", "gateName", "gateArgs", "qubit", "measureStmt", 
      "barrierStmt", "qubitList"
    },
    std::vector<std::string>{
      "", "'OPENQASM'", "'.'", "'include'", "'qreg'", "'creg'", "'pi'", 
      "'+'", "'-'", "'*'", "'/'", "'->'", "'gphase'", "'id'", "", "'x'", 
      "'y'", "'z'", "'sx'", "'h'", "'s'", "'sdg'", "'t'", "'tdg'", "'rx'", 
      "'ry'", "'rz'", "", "'cy'", "'cz'", "", "'crx'", "'cry'", "'crz'", 
      "'ch'", "'cu'", "'swap'", "'ccx'", "'cswap'", "'u1'", "'u2'", "'u3'", 
      "'reset'", "'barrier'", "'measure'", "", "", "", "'['", "']'", "'('", 
      "')'", "','", "';'"
    },
    std::vector<std::string>{
      "", "", "", "", "", "", "", "", "", "", "", "", "GPHASE", "ID", "P", 
      "X", "Y", "Z", "SX", "H", "S", "SDG", "T", "TDG", "RX", "RY", "RZ", 
      "CX", "CY", "CZ", "CP", "CRX", "CRY", "CRZ", "CH", "CU", "SWAP", "CCX", 
      "CSWAP", "U1", "U2", "U3", "RESET", "BARRIER", "MEASURE", "NUMBER", 
      "IDSTR", "STRING", "LBRACKET", "RBRACKET", "LPAREN", "RPAREN", "COMMA", 
      "SEMICOLON", "WS"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,54,175,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,1,0,1,0,
  	1,0,1,0,1,0,1,0,5,0,35,8,0,10,0,12,0,38,9,0,1,0,5,0,41,8,0,10,0,12,0,
  	44,9,0,1,1,1,1,1,1,1,1,1,2,1,2,1,2,1,2,1,2,3,2,55,8,2,1,3,1,3,1,3,1,3,
  	1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,5,3,5,74,8,5,1,5,3,
  	5,77,8,5,1,5,1,5,1,5,1,6,1,6,1,6,5,6,85,8,6,10,6,12,6,88,9,6,1,7,1,7,
  	1,7,1,7,1,7,1,7,1,7,1,7,3,7,98,8,7,1,7,1,7,1,7,5,7,103,8,7,10,7,12,7,
  	106,9,7,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,
  	8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,3,8,139,
  	8,8,1,9,1,9,1,9,5,9,144,8,9,10,9,12,9,147,9,9,1,10,1,10,1,10,1,10,1,10,
  	1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,12,1,12,1,12,1,12,1,13,
  	1,13,1,13,5,13,170,8,13,10,13,12,13,173,9,13,1,13,0,1,14,14,0,2,4,6,8,
  	10,12,14,16,18,20,22,24,26,0,1,1,0,7,10,205,0,28,1,0,0,0,2,45,1,0,0,0,
  	4,54,1,0,0,0,6,56,1,0,0,0,8,63,1,0,0,0,10,70,1,0,0,0,12,81,1,0,0,0,14,
  	97,1,0,0,0,16,138,1,0,0,0,18,140,1,0,0,0,20,148,1,0,0,0,22,153,1,0,0,
  	0,24,162,1,0,0,0,26,166,1,0,0,0,28,29,5,1,0,0,29,30,5,45,0,0,30,31,5,
  	2,0,0,31,32,5,45,0,0,32,36,5,53,0,0,33,35,3,2,1,0,34,33,1,0,0,0,35,38,
  	1,0,0,0,36,34,1,0,0,0,36,37,1,0,0,0,37,42,1,0,0,0,38,36,1,0,0,0,39,41,
  	3,4,2,0,40,39,1,0,0,0,41,44,1,0,0,0,42,40,1,0,0,0,42,43,1,0,0,0,43,1,
  	1,0,0,0,44,42,1,0,0,0,45,46,5,3,0,0,46,47,5,47,0,0,47,48,5,53,0,0,48,
  	3,1,0,0,0,49,55,3,6,3,0,50,55,3,8,4,0,51,55,3,10,5,0,52,55,3,22,11,0,
  	53,55,3,24,12,0,54,49,1,0,0,0,54,50,1,0,0,0,54,51,1,0,0,0,54,52,1,0,0,
  	0,54,53,1,0,0,0,55,5,1,0,0,0,56,57,5,4,0,0,57,58,5,46,0,0,58,59,5,48,
  	0,0,59,60,5,45,0,0,60,61,5,49,0,0,61,62,5,53,0,0,62,7,1,0,0,0,63,64,5,
  	5,0,0,64,65,5,46,0,0,65,66,5,48,0,0,66,67,5,45,0,0,67,68,5,49,0,0,68,
  	69,5,53,0,0,69,9,1,0,0,0,70,76,3,16,8,0,71,73,5,50,0,0,72,74,3,12,6,0,
  	73,72,1,0,0,0,73,74,1,0,0,0,74,75,1,0,0,0,75,77,5,51,0,0,76,71,1,0,0,
  	0,76,77,1,0,0,0,77,78,1,0,0,0,78,79,3,18,9,0,79,80,5,53,0,0,80,11,1,0,
  	0,0,81,86,3,14,7,0,82,83,5,52,0,0,83,85,3,14,7,0,84,82,1,0,0,0,85,88,
  	1,0,0,0,86,84,1,0,0,0,86,87,1,0,0,0,87,13,1,0,0,0,88,86,1,0,0,0,89,90,
  	6,7,-1,0,90,98,5,45,0,0,91,98,5,46,0,0,92,98,5,6,0,0,93,94,5,50,0,0,94,
  	95,3,14,7,0,95,96,5,51,0,0,96,98,1,0,0,0,97,89,1,0,0,0,97,91,1,0,0,0,
  	97,92,1,0,0,0,97,93,1,0,0,0,98,104,1,0,0,0,99,100,10,2,0,0,100,101,7,
  	0,0,0,101,103,3,14,7,3,102,99,1,0,0,0,103,106,1,0,0,0,104,102,1,0,0,0,
  	104,105,1,0,0,0,105,15,1,0,0,0,106,104,1,0,0,0,107,139,5,15,0,0,108,139,
  	5,16,0,0,109,139,5,17,0,0,110,139,5,19,0,0,111,139,5,20,0,0,112,139,5,
  	21,0,0,113,139,5,22,0,0,114,139,5,23,0,0,115,139,5,24,0,0,116,139,5,25,
  	0,0,117,139,5,26,0,0,118,139,5,27,0,0,119,139,5,28,0,0,120,139,5,29,0,
  	0,121,139,5,30,0,0,122,139,5,36,0,0,123,139,5,37,0,0,124,139,5,38,0,0,
  	125,139,5,39,0,0,126,139,5,40,0,0,127,139,5,41,0,0,128,139,5,18,0,0,129,
  	139,5,34,0,0,130,139,5,31,0,0,131,139,5,32,0,0,132,139,5,33,0,0,133,139,
  	5,35,0,0,134,139,5,14,0,0,135,139,5,13,0,0,136,139,5,12,0,0,137,139,1,
  	0,0,0,138,107,1,0,0,0,138,108,1,0,0,0,138,109,1,0,0,0,138,110,1,0,0,0,
  	138,111,1,0,0,0,138,112,1,0,0,0,138,113,1,0,0,0,138,114,1,0,0,0,138,115,
  	1,0,0,0,138,116,1,0,0,0,138,117,1,0,0,0,138,118,1,0,0,0,138,119,1,0,0,
  	0,138,120,1,0,0,0,138,121,1,0,0,0,138,122,1,0,0,0,138,123,1,0,0,0,138,
  	124,1,0,0,0,138,125,1,0,0,0,138,126,1,0,0,0,138,127,1,0,0,0,138,128,1,
  	0,0,0,138,129,1,0,0,0,138,130,1,0,0,0,138,131,1,0,0,0,138,132,1,0,0,0,
  	138,133,1,0,0,0,138,134,1,0,0,0,138,135,1,0,0,0,138,136,1,0,0,0,138,137,
  	1,0,0,0,139,17,1,0,0,0,140,145,3,20,10,0,141,142,5,52,0,0,142,144,3,20,
  	10,0,143,141,1,0,0,0,144,147,1,0,0,0,145,143,1,0,0,0,145,146,1,0,0,0,
  	146,19,1,0,0,0,147,145,1,0,0,0,148,149,5,46,0,0,149,150,5,48,0,0,150,
  	151,5,45,0,0,151,152,5,49,0,0,152,21,1,0,0,0,153,154,5,44,0,0,154,155,
  	3,20,10,0,155,156,5,11,0,0,156,157,5,46,0,0,157,158,5,48,0,0,158,159,
  	5,45,0,0,159,160,5,49,0,0,160,161,5,53,0,0,161,23,1,0,0,0,162,163,5,43,
  	0,0,163,164,3,26,13,0,164,165,5,53,0,0,165,25,1,0,0,0,166,171,3,20,10,
  	0,167,168,5,52,0,0,168,170,3,20,10,0,169,167,1,0,0,0,170,173,1,0,0,0,
  	171,169,1,0,0,0,171,172,1,0,0,0,172,27,1,0,0,0,173,171,1,0,0,0,11,36,
  	42,54,73,76,86,97,104,138,145,171
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  openqasm3ParserStaticData = std::move(staticData);
}

}

OpenQASM3Parser::OpenQASM3Parser(TokenStream *input) : OpenQASM3Parser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

OpenQASM3Parser::OpenQASM3Parser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  OpenQASM3Parser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *openqasm3ParserStaticData->atn, openqasm3ParserStaticData->decisionToDFA, openqasm3ParserStaticData->sharedContextCache, options);
}

OpenQASM3Parser::~OpenQASM3Parser() {
  delete _interpreter;
}

const atn::ATN& OpenQASM3Parser::getATN() const {
  return *openqasm3ParserStaticData->atn;
}

std::string OpenQASM3Parser::getGrammarFileName() const {
  return "OpenQASM3.g4";
}

const std::vector<std::string>& OpenQASM3Parser::getRuleNames() const {
  return openqasm3ParserStaticData->ruleNames;
}

const dfa::Vocabulary& OpenQASM3Parser::getVocabulary() const {
  return openqasm3ParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView OpenQASM3Parser::getSerializedATN() const {
  return openqasm3ParserStaticData->serializedATN;
}


//----------------- ProgramContext ------------------------------------------------------------------

OpenQASM3Parser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> OpenQASM3Parser::ProgramContext::NUMBER() {
  return getTokens(OpenQASM3Parser::NUMBER);
}

tree::TerminalNode* OpenQASM3Parser::ProgramContext::NUMBER(size_t i) {
  return getToken(OpenQASM3Parser::NUMBER, i);
}

tree::TerminalNode* OpenQASM3Parser::ProgramContext::SEMICOLON() {
  return getToken(OpenQASM3Parser::SEMICOLON, 0);
}

std::vector<OpenQASM3Parser::IncludeStmtContext *> OpenQASM3Parser::ProgramContext::includeStmt() {
  return getRuleContexts<OpenQASM3Parser::IncludeStmtContext>();
}

OpenQASM3Parser::IncludeStmtContext* OpenQASM3Parser::ProgramContext::includeStmt(size_t i) {
  return getRuleContext<OpenQASM3Parser::IncludeStmtContext>(i);
}

std::vector<OpenQASM3Parser::StatementContext *> OpenQASM3Parser::ProgramContext::statement() {
  return getRuleContexts<OpenQASM3Parser::StatementContext>();
}

OpenQASM3Parser::StatementContext* OpenQASM3Parser::ProgramContext::statement(size_t i) {
  return getRuleContext<OpenQASM3Parser::StatementContext>(i);
}


size_t OpenQASM3Parser::ProgramContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleProgram;
}


std::any OpenQASM3Parser::ProgramContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitProgram(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::ProgramContext* OpenQASM3Parser::program() {
  ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, getState());
  enterRule(_localctx, 0, OpenQASM3Parser::RuleProgram);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(28);
    match(OpenQASM3Parser::T__0);
    setState(29);
    match(OpenQASM3Parser::NUMBER);
    setState(30);
    match(OpenQASM3Parser::T__1);
    setState(31);
    match(OpenQASM3Parser::NUMBER);
    setState(32);
    match(OpenQASM3Parser::SEMICOLON);
    setState(36);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == OpenQASM3Parser::T__2) {
      setState(33);
      includeStmt();
      setState(38);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(42);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 1227054976593968) != 0)) {
      setState(39);
      statement();
      setState(44);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IncludeStmtContext ------------------------------------------------------------------

OpenQASM3Parser::IncludeStmtContext::IncludeStmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* OpenQASM3Parser::IncludeStmtContext::STRING() {
  return getToken(OpenQASM3Parser::STRING, 0);
}

tree::TerminalNode* OpenQASM3Parser::IncludeStmtContext::SEMICOLON() {
  return getToken(OpenQASM3Parser::SEMICOLON, 0);
}


size_t OpenQASM3Parser::IncludeStmtContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleIncludeStmt;
}


std::any OpenQASM3Parser::IncludeStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitIncludeStmt(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::IncludeStmtContext* OpenQASM3Parser::includeStmt() {
  IncludeStmtContext *_localctx = _tracker.createInstance<IncludeStmtContext>(_ctx, getState());
  enterRule(_localctx, 2, OpenQASM3Parser::RuleIncludeStmt);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(45);
    match(OpenQASM3Parser::T__2);
    setState(46);
    match(OpenQASM3Parser::STRING);
    setState(47);
    match(OpenQASM3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

OpenQASM3Parser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

OpenQASM3Parser::QregDeclContext* OpenQASM3Parser::StatementContext::qregDecl() {
  return getRuleContext<OpenQASM3Parser::QregDeclContext>(0);
}

OpenQASM3Parser::CregDeclContext* OpenQASM3Parser::StatementContext::cregDecl() {
  return getRuleContext<OpenQASM3Parser::CregDeclContext>(0);
}

OpenQASM3Parser::GateStmtContext* OpenQASM3Parser::StatementContext::gateStmt() {
  return getRuleContext<OpenQASM3Parser::GateStmtContext>(0);
}

OpenQASM3Parser::MeasureStmtContext* OpenQASM3Parser::StatementContext::measureStmt() {
  return getRuleContext<OpenQASM3Parser::MeasureStmtContext>(0);
}

OpenQASM3Parser::BarrierStmtContext* OpenQASM3Parser::StatementContext::barrierStmt() {
  return getRuleContext<OpenQASM3Parser::BarrierStmtContext>(0);
}


size_t OpenQASM3Parser::StatementContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleStatement;
}


std::any OpenQASM3Parser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::StatementContext* OpenQASM3Parser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 4, OpenQASM3Parser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(54);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case OpenQASM3Parser::T__3: {
        enterOuterAlt(_localctx, 1);
        setState(49);
        qregDecl();
        break;
      }

      case OpenQASM3Parser::T__4: {
        enterOuterAlt(_localctx, 2);
        setState(50);
        cregDecl();
        break;
      }

      case OpenQASM3Parser::GPHASE:
      case OpenQASM3Parser::ID:
      case OpenQASM3Parser::P:
      case OpenQASM3Parser::X:
      case OpenQASM3Parser::Y:
      case OpenQASM3Parser::Z:
      case OpenQASM3Parser::SX:
      case OpenQASM3Parser::H:
      case OpenQASM3Parser::S:
      case OpenQASM3Parser::SDG:
      case OpenQASM3Parser::T:
      case OpenQASM3Parser::TDG:
      case OpenQASM3Parser::RX:
      case OpenQASM3Parser::RY:
      case OpenQASM3Parser::RZ:
      case OpenQASM3Parser::CX:
      case OpenQASM3Parser::CY:
      case OpenQASM3Parser::CZ:
      case OpenQASM3Parser::CP:
      case OpenQASM3Parser::CRX:
      case OpenQASM3Parser::CRY:
      case OpenQASM3Parser::CRZ:
      case OpenQASM3Parser::CH:
      case OpenQASM3Parser::CU:
      case OpenQASM3Parser::SWAP:
      case OpenQASM3Parser::CCX:
      case OpenQASM3Parser::CSWAP:
      case OpenQASM3Parser::U1:
      case OpenQASM3Parser::U2:
      case OpenQASM3Parser::U3:
      case OpenQASM3Parser::IDSTR:
      case OpenQASM3Parser::LPAREN: {
        enterOuterAlt(_localctx, 3);
        setState(51);
        gateStmt();
        break;
      }

      case OpenQASM3Parser::MEASURE: {
        enterOuterAlt(_localctx, 4);
        setState(52);
        measureStmt();
        break;
      }

      case OpenQASM3Parser::BARRIER: {
        enterOuterAlt(_localctx, 5);
        setState(53);
        barrierStmt();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QregDeclContext ------------------------------------------------------------------

OpenQASM3Parser::QregDeclContext::QregDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* OpenQASM3Parser::QregDeclContext::IDSTR() {
  return getToken(OpenQASM3Parser::IDSTR, 0);
}

tree::TerminalNode* OpenQASM3Parser::QregDeclContext::LBRACKET() {
  return getToken(OpenQASM3Parser::LBRACKET, 0);
}

tree::TerminalNode* OpenQASM3Parser::QregDeclContext::NUMBER() {
  return getToken(OpenQASM3Parser::NUMBER, 0);
}

tree::TerminalNode* OpenQASM3Parser::QregDeclContext::RBRACKET() {
  return getToken(OpenQASM3Parser::RBRACKET, 0);
}

tree::TerminalNode* OpenQASM3Parser::QregDeclContext::SEMICOLON() {
  return getToken(OpenQASM3Parser::SEMICOLON, 0);
}


size_t OpenQASM3Parser::QregDeclContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleQregDecl;
}


std::any OpenQASM3Parser::QregDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitQregDecl(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::QregDeclContext* OpenQASM3Parser::qregDecl() {
  QregDeclContext *_localctx = _tracker.createInstance<QregDeclContext>(_ctx, getState());
  enterRule(_localctx, 6, OpenQASM3Parser::RuleQregDecl);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(56);
    match(OpenQASM3Parser::T__3);
    setState(57);
    match(OpenQASM3Parser::IDSTR);
    setState(58);
    match(OpenQASM3Parser::LBRACKET);
    setState(59);
    match(OpenQASM3Parser::NUMBER);
    setState(60);
    match(OpenQASM3Parser::RBRACKET);
    setState(61);
    match(OpenQASM3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CregDeclContext ------------------------------------------------------------------

OpenQASM3Parser::CregDeclContext::CregDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* OpenQASM3Parser::CregDeclContext::IDSTR() {
  return getToken(OpenQASM3Parser::IDSTR, 0);
}

tree::TerminalNode* OpenQASM3Parser::CregDeclContext::LBRACKET() {
  return getToken(OpenQASM3Parser::LBRACKET, 0);
}

tree::TerminalNode* OpenQASM3Parser::CregDeclContext::NUMBER() {
  return getToken(OpenQASM3Parser::NUMBER, 0);
}

tree::TerminalNode* OpenQASM3Parser::CregDeclContext::RBRACKET() {
  return getToken(OpenQASM3Parser::RBRACKET, 0);
}

tree::TerminalNode* OpenQASM3Parser::CregDeclContext::SEMICOLON() {
  return getToken(OpenQASM3Parser::SEMICOLON, 0);
}


size_t OpenQASM3Parser::CregDeclContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleCregDecl;
}


std::any OpenQASM3Parser::CregDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitCregDecl(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::CregDeclContext* OpenQASM3Parser::cregDecl() {
  CregDeclContext *_localctx = _tracker.createInstance<CregDeclContext>(_ctx, getState());
  enterRule(_localctx, 8, OpenQASM3Parser::RuleCregDecl);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(63);
    match(OpenQASM3Parser::T__4);
    setState(64);
    match(OpenQASM3Parser::IDSTR);
    setState(65);
    match(OpenQASM3Parser::LBRACKET);
    setState(66);
    match(OpenQASM3Parser::NUMBER);
    setState(67);
    match(OpenQASM3Parser::RBRACKET);
    setState(68);
    match(OpenQASM3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GateStmtContext ------------------------------------------------------------------

OpenQASM3Parser::GateStmtContext::GateStmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

OpenQASM3Parser::GateNameContext* OpenQASM3Parser::GateStmtContext::gateName() {
  return getRuleContext<OpenQASM3Parser::GateNameContext>(0);
}

OpenQASM3Parser::GateArgsContext* OpenQASM3Parser::GateStmtContext::gateArgs() {
  return getRuleContext<OpenQASM3Parser::GateArgsContext>(0);
}

tree::TerminalNode* OpenQASM3Parser::GateStmtContext::SEMICOLON() {
  return getToken(OpenQASM3Parser::SEMICOLON, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateStmtContext::LPAREN() {
  return getToken(OpenQASM3Parser::LPAREN, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateStmtContext::RPAREN() {
  return getToken(OpenQASM3Parser::RPAREN, 0);
}

OpenQASM3Parser::ParamListContext* OpenQASM3Parser::GateStmtContext::paramList() {
  return getRuleContext<OpenQASM3Parser::ParamListContext>(0);
}


size_t OpenQASM3Parser::GateStmtContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleGateStmt;
}


std::any OpenQASM3Parser::GateStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitGateStmt(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::GateStmtContext* OpenQASM3Parser::gateStmt() {
  GateStmtContext *_localctx = _tracker.createInstance<GateStmtContext>(_ctx, getState());
  enterRule(_localctx, 10, OpenQASM3Parser::RuleGateStmt);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(70);
    gateName();
    setState(76);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == OpenQASM3Parser::LPAREN) {
      setState(71);
      match(OpenQASM3Parser::LPAREN);
      setState(73);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 1231453023109184) != 0)) {
        setState(72);
        paramList();
      }
      setState(75);
      match(OpenQASM3Parser::RPAREN);
    }
    setState(78);
    gateArgs();
    setState(79);
    match(OpenQASM3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ParamListContext ------------------------------------------------------------------

OpenQASM3Parser::ParamListContext::ParamListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<OpenQASM3Parser::ExprContext *> OpenQASM3Parser::ParamListContext::expr() {
  return getRuleContexts<OpenQASM3Parser::ExprContext>();
}

OpenQASM3Parser::ExprContext* OpenQASM3Parser::ParamListContext::expr(size_t i) {
  return getRuleContext<OpenQASM3Parser::ExprContext>(i);
}

std::vector<tree::TerminalNode *> OpenQASM3Parser::ParamListContext::COMMA() {
  return getTokens(OpenQASM3Parser::COMMA);
}

tree::TerminalNode* OpenQASM3Parser::ParamListContext::COMMA(size_t i) {
  return getToken(OpenQASM3Parser::COMMA, i);
}


size_t OpenQASM3Parser::ParamListContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleParamList;
}


std::any OpenQASM3Parser::ParamListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitParamList(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::ParamListContext* OpenQASM3Parser::paramList() {
  ParamListContext *_localctx = _tracker.createInstance<ParamListContext>(_ctx, getState());
  enterRule(_localctx, 12, OpenQASM3Parser::RuleParamList);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(81);
    expr(0);
    setState(86);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == OpenQASM3Parser::COMMA) {
      setState(82);
      match(OpenQASM3Parser::COMMA);
      setState(83);
      expr(0);
      setState(88);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExprContext ------------------------------------------------------------------

OpenQASM3Parser::ExprContext::ExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* OpenQASM3Parser::ExprContext::NUMBER() {
  return getToken(OpenQASM3Parser::NUMBER, 0);
}

tree::TerminalNode* OpenQASM3Parser::ExprContext::IDSTR() {
  return getToken(OpenQASM3Parser::IDSTR, 0);
}

tree::TerminalNode* OpenQASM3Parser::ExprContext::LPAREN() {
  return getToken(OpenQASM3Parser::LPAREN, 0);
}

std::vector<OpenQASM3Parser::ExprContext *> OpenQASM3Parser::ExprContext::expr() {
  return getRuleContexts<OpenQASM3Parser::ExprContext>();
}

OpenQASM3Parser::ExprContext* OpenQASM3Parser::ExprContext::expr(size_t i) {
  return getRuleContext<OpenQASM3Parser::ExprContext>(i);
}

tree::TerminalNode* OpenQASM3Parser::ExprContext::RPAREN() {
  return getToken(OpenQASM3Parser::RPAREN, 0);
}


size_t OpenQASM3Parser::ExprContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleExpr;
}


std::any OpenQASM3Parser::ExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitExpr(this);
  else
    return visitor->visitChildren(this);
}


OpenQASM3Parser::ExprContext* OpenQASM3Parser::expr() {
   return expr(0);
}

OpenQASM3Parser::ExprContext* OpenQASM3Parser::expr(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  OpenQASM3Parser::ExprContext *_localctx = _tracker.createInstance<ExprContext>(_ctx, parentState);
  OpenQASM3Parser::ExprContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 14;
  enterRecursionRule(_localctx, 14, OpenQASM3Parser::RuleExpr, precedence);

    size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(97);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case OpenQASM3Parser::NUMBER: {
        setState(90);
        match(OpenQASM3Parser::NUMBER);
        break;
      }

      case OpenQASM3Parser::IDSTR: {
        setState(91);
        match(OpenQASM3Parser::IDSTR);
        break;
      }

      case OpenQASM3Parser::T__5: {
        setState(92);
        match(OpenQASM3Parser::T__5);
        break;
      }

      case OpenQASM3Parser::LPAREN: {
        setState(93);
        match(OpenQASM3Parser::LPAREN);
        setState(94);
        expr(0);
        setState(95);
        match(OpenQASM3Parser::RPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(104);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleExpr);
        setState(99);

        if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
        setState(100);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & 1920) != 0))) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(101);
        expr(3); 
      }
      setState(106);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- GateNameContext ------------------------------------------------------------------

OpenQASM3Parser::GateNameContext::GateNameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::X() {
  return getToken(OpenQASM3Parser::X, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::Y() {
  return getToken(OpenQASM3Parser::Y, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::Z() {
  return getToken(OpenQASM3Parser::Z, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::H() {
  return getToken(OpenQASM3Parser::H, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::S() {
  return getToken(OpenQASM3Parser::S, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::SDG() {
  return getToken(OpenQASM3Parser::SDG, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::T() {
  return getToken(OpenQASM3Parser::T, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::TDG() {
  return getToken(OpenQASM3Parser::TDG, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::RX() {
  return getToken(OpenQASM3Parser::RX, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::RY() {
  return getToken(OpenQASM3Parser::RY, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::RZ() {
  return getToken(OpenQASM3Parser::RZ, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CX() {
  return getToken(OpenQASM3Parser::CX, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CY() {
  return getToken(OpenQASM3Parser::CY, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CZ() {
  return getToken(OpenQASM3Parser::CZ, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CP() {
  return getToken(OpenQASM3Parser::CP, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::SWAP() {
  return getToken(OpenQASM3Parser::SWAP, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CCX() {
  return getToken(OpenQASM3Parser::CCX, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CSWAP() {
  return getToken(OpenQASM3Parser::CSWAP, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::U1() {
  return getToken(OpenQASM3Parser::U1, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::U2() {
  return getToken(OpenQASM3Parser::U2, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::U3() {
  return getToken(OpenQASM3Parser::U3, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::SX() {
  return getToken(OpenQASM3Parser::SX, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CH() {
  return getToken(OpenQASM3Parser::CH, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CRX() {
  return getToken(OpenQASM3Parser::CRX, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CRY() {
  return getToken(OpenQASM3Parser::CRY, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CRZ() {
  return getToken(OpenQASM3Parser::CRZ, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::CU() {
  return getToken(OpenQASM3Parser::CU, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::P() {
  return getToken(OpenQASM3Parser::P, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::ID() {
  return getToken(OpenQASM3Parser::ID, 0);
}

tree::TerminalNode* OpenQASM3Parser::GateNameContext::GPHASE() {
  return getToken(OpenQASM3Parser::GPHASE, 0);
}


size_t OpenQASM3Parser::GateNameContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleGateName;
}


std::any OpenQASM3Parser::GateNameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitGateName(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::GateNameContext* OpenQASM3Parser::gateName() {
  GateNameContext *_localctx = _tracker.createInstance<GateNameContext>(_ctx, getState());
  enterRule(_localctx, 16, OpenQASM3Parser::RuleGateName);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(138);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case OpenQASM3Parser::X: {
        enterOuterAlt(_localctx, 1);
        setState(107);
        match(OpenQASM3Parser::X);
        break;
      }

      case OpenQASM3Parser::Y: {
        enterOuterAlt(_localctx, 2);
        setState(108);
        match(OpenQASM3Parser::Y);
        break;
      }

      case OpenQASM3Parser::Z: {
        enterOuterAlt(_localctx, 3);
        setState(109);
        match(OpenQASM3Parser::Z);
        break;
      }

      case OpenQASM3Parser::H: {
        enterOuterAlt(_localctx, 4);
        setState(110);
        match(OpenQASM3Parser::H);
        break;
      }

      case OpenQASM3Parser::S: {
        enterOuterAlt(_localctx, 5);
        setState(111);
        match(OpenQASM3Parser::S);
        break;
      }

      case OpenQASM3Parser::SDG: {
        enterOuterAlt(_localctx, 6);
        setState(112);
        match(OpenQASM3Parser::SDG);
        break;
      }

      case OpenQASM3Parser::T: {
        enterOuterAlt(_localctx, 7);
        setState(113);
        match(OpenQASM3Parser::T);
        break;
      }

      case OpenQASM3Parser::TDG: {
        enterOuterAlt(_localctx, 8);
        setState(114);
        match(OpenQASM3Parser::TDG);
        break;
      }

      case OpenQASM3Parser::RX: {
        enterOuterAlt(_localctx, 9);
        setState(115);
        match(OpenQASM3Parser::RX);
        break;
      }

      case OpenQASM3Parser::RY: {
        enterOuterAlt(_localctx, 10);
        setState(116);
        match(OpenQASM3Parser::RY);
        break;
      }

      case OpenQASM3Parser::RZ: {
        enterOuterAlt(_localctx, 11);
        setState(117);
        match(OpenQASM3Parser::RZ);
        break;
      }

      case OpenQASM3Parser::CX: {
        enterOuterAlt(_localctx, 12);
        setState(118);
        match(OpenQASM3Parser::CX);
        break;
      }

      case OpenQASM3Parser::CY: {
        enterOuterAlt(_localctx, 13);
        setState(119);
        match(OpenQASM3Parser::CY);
        break;
      }

      case OpenQASM3Parser::CZ: {
        enterOuterAlt(_localctx, 14);
        setState(120);
        match(OpenQASM3Parser::CZ);
        break;
      }

      case OpenQASM3Parser::CP: {
        enterOuterAlt(_localctx, 15);
        setState(121);
        match(OpenQASM3Parser::CP);
        break;
      }

      case OpenQASM3Parser::SWAP: {
        enterOuterAlt(_localctx, 16);
        setState(122);
        match(OpenQASM3Parser::SWAP);
        break;
      }

      case OpenQASM3Parser::CCX: {
        enterOuterAlt(_localctx, 17);
        setState(123);
        match(OpenQASM3Parser::CCX);
        break;
      }

      case OpenQASM3Parser::CSWAP: {
        enterOuterAlt(_localctx, 18);
        setState(124);
        match(OpenQASM3Parser::CSWAP);
        break;
      }

      case OpenQASM3Parser::U1: {
        enterOuterAlt(_localctx, 19);
        setState(125);
        match(OpenQASM3Parser::U1);
        break;
      }

      case OpenQASM3Parser::U2: {
        enterOuterAlt(_localctx, 20);
        setState(126);
        match(OpenQASM3Parser::U2);
        break;
      }

      case OpenQASM3Parser::U3: {
        enterOuterAlt(_localctx, 21);
        setState(127);
        match(OpenQASM3Parser::U3);
        break;
      }

      case OpenQASM3Parser::SX: {
        enterOuterAlt(_localctx, 22);
        setState(128);
        match(OpenQASM3Parser::SX);
        break;
      }

      case OpenQASM3Parser::CH: {
        enterOuterAlt(_localctx, 23);
        setState(129);
        match(OpenQASM3Parser::CH);
        break;
      }

      case OpenQASM3Parser::CRX: {
        enterOuterAlt(_localctx, 24);
        setState(130);
        match(OpenQASM3Parser::CRX);
        break;
      }

      case OpenQASM3Parser::CRY: {
        enterOuterAlt(_localctx, 25);
        setState(131);
        match(OpenQASM3Parser::CRY);
        break;
      }

      case OpenQASM3Parser::CRZ: {
        enterOuterAlt(_localctx, 26);
        setState(132);
        match(OpenQASM3Parser::CRZ);
        break;
      }

      case OpenQASM3Parser::CU: {
        enterOuterAlt(_localctx, 27);
        setState(133);
        match(OpenQASM3Parser::CU);
        break;
      }

      case OpenQASM3Parser::P: {
        enterOuterAlt(_localctx, 28);
        setState(134);
        match(OpenQASM3Parser::P);
        break;
      }

      case OpenQASM3Parser::ID: {
        enterOuterAlt(_localctx, 29);
        setState(135);
        match(OpenQASM3Parser::ID);
        break;
      }

      case OpenQASM3Parser::GPHASE: {
        enterOuterAlt(_localctx, 30);
        setState(136);
        match(OpenQASM3Parser::GPHASE);
        break;
      }

      case OpenQASM3Parser::IDSTR:
      case OpenQASM3Parser::LPAREN: {
        enterOuterAlt(_localctx, 31);

        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GateArgsContext ------------------------------------------------------------------

OpenQASM3Parser::GateArgsContext::GateArgsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<OpenQASM3Parser::QubitContext *> OpenQASM3Parser::GateArgsContext::qubit() {
  return getRuleContexts<OpenQASM3Parser::QubitContext>();
}

OpenQASM3Parser::QubitContext* OpenQASM3Parser::GateArgsContext::qubit(size_t i) {
  return getRuleContext<OpenQASM3Parser::QubitContext>(i);
}

std::vector<tree::TerminalNode *> OpenQASM3Parser::GateArgsContext::COMMA() {
  return getTokens(OpenQASM3Parser::COMMA);
}

tree::TerminalNode* OpenQASM3Parser::GateArgsContext::COMMA(size_t i) {
  return getToken(OpenQASM3Parser::COMMA, i);
}


size_t OpenQASM3Parser::GateArgsContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleGateArgs;
}


std::any OpenQASM3Parser::GateArgsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitGateArgs(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::GateArgsContext* OpenQASM3Parser::gateArgs() {
  GateArgsContext *_localctx = _tracker.createInstance<GateArgsContext>(_ctx, getState());
  enterRule(_localctx, 18, OpenQASM3Parser::RuleGateArgs);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(140);
    qubit();
    setState(145);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == OpenQASM3Parser::COMMA) {
      setState(141);
      match(OpenQASM3Parser::COMMA);
      setState(142);
      qubit();
      setState(147);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QubitContext ------------------------------------------------------------------

OpenQASM3Parser::QubitContext::QubitContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* OpenQASM3Parser::QubitContext::IDSTR() {
  return getToken(OpenQASM3Parser::IDSTR, 0);
}

tree::TerminalNode* OpenQASM3Parser::QubitContext::LBRACKET() {
  return getToken(OpenQASM3Parser::LBRACKET, 0);
}

tree::TerminalNode* OpenQASM3Parser::QubitContext::NUMBER() {
  return getToken(OpenQASM3Parser::NUMBER, 0);
}

tree::TerminalNode* OpenQASM3Parser::QubitContext::RBRACKET() {
  return getToken(OpenQASM3Parser::RBRACKET, 0);
}


size_t OpenQASM3Parser::QubitContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleQubit;
}


std::any OpenQASM3Parser::QubitContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitQubit(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::QubitContext* OpenQASM3Parser::qubit() {
  QubitContext *_localctx = _tracker.createInstance<QubitContext>(_ctx, getState());
  enterRule(_localctx, 20, OpenQASM3Parser::RuleQubit);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(148);
    match(OpenQASM3Parser::IDSTR);
    setState(149);
    match(OpenQASM3Parser::LBRACKET);
    setState(150);
    match(OpenQASM3Parser::NUMBER);
    setState(151);
    match(OpenQASM3Parser::RBRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MeasureStmtContext ------------------------------------------------------------------

OpenQASM3Parser::MeasureStmtContext::MeasureStmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* OpenQASM3Parser::MeasureStmtContext::MEASURE() {
  return getToken(OpenQASM3Parser::MEASURE, 0);
}

OpenQASM3Parser::QubitContext* OpenQASM3Parser::MeasureStmtContext::qubit() {
  return getRuleContext<OpenQASM3Parser::QubitContext>(0);
}

tree::TerminalNode* OpenQASM3Parser::MeasureStmtContext::IDSTR() {
  return getToken(OpenQASM3Parser::IDSTR, 0);
}

tree::TerminalNode* OpenQASM3Parser::MeasureStmtContext::LBRACKET() {
  return getToken(OpenQASM3Parser::LBRACKET, 0);
}

tree::TerminalNode* OpenQASM3Parser::MeasureStmtContext::NUMBER() {
  return getToken(OpenQASM3Parser::NUMBER, 0);
}

tree::TerminalNode* OpenQASM3Parser::MeasureStmtContext::RBRACKET() {
  return getToken(OpenQASM3Parser::RBRACKET, 0);
}

tree::TerminalNode* OpenQASM3Parser::MeasureStmtContext::SEMICOLON() {
  return getToken(OpenQASM3Parser::SEMICOLON, 0);
}


size_t OpenQASM3Parser::MeasureStmtContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleMeasureStmt;
}


std::any OpenQASM3Parser::MeasureStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitMeasureStmt(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::MeasureStmtContext* OpenQASM3Parser::measureStmt() {
  MeasureStmtContext *_localctx = _tracker.createInstance<MeasureStmtContext>(_ctx, getState());
  enterRule(_localctx, 22, OpenQASM3Parser::RuleMeasureStmt);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(153);
    match(OpenQASM3Parser::MEASURE);
    setState(154);
    qubit();
    setState(155);
    match(OpenQASM3Parser::T__10);
    setState(156);
    match(OpenQASM3Parser::IDSTR);
    setState(157);
    match(OpenQASM3Parser::LBRACKET);
    setState(158);
    match(OpenQASM3Parser::NUMBER);
    setState(159);
    match(OpenQASM3Parser::RBRACKET);
    setState(160);
    match(OpenQASM3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BarrierStmtContext ------------------------------------------------------------------

OpenQASM3Parser::BarrierStmtContext::BarrierStmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* OpenQASM3Parser::BarrierStmtContext::BARRIER() {
  return getToken(OpenQASM3Parser::BARRIER, 0);
}

OpenQASM3Parser::QubitListContext* OpenQASM3Parser::BarrierStmtContext::qubitList() {
  return getRuleContext<OpenQASM3Parser::QubitListContext>(0);
}

tree::TerminalNode* OpenQASM3Parser::BarrierStmtContext::SEMICOLON() {
  return getToken(OpenQASM3Parser::SEMICOLON, 0);
}


size_t OpenQASM3Parser::BarrierStmtContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleBarrierStmt;
}


std::any OpenQASM3Parser::BarrierStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitBarrierStmt(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::BarrierStmtContext* OpenQASM3Parser::barrierStmt() {
  BarrierStmtContext *_localctx = _tracker.createInstance<BarrierStmtContext>(_ctx, getState());
  enterRule(_localctx, 24, OpenQASM3Parser::RuleBarrierStmt);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(162);
    match(OpenQASM3Parser::BARRIER);
    setState(163);
    qubitList();
    setState(164);
    match(OpenQASM3Parser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QubitListContext ------------------------------------------------------------------

OpenQASM3Parser::QubitListContext::QubitListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<OpenQASM3Parser::QubitContext *> OpenQASM3Parser::QubitListContext::qubit() {
  return getRuleContexts<OpenQASM3Parser::QubitContext>();
}

OpenQASM3Parser::QubitContext* OpenQASM3Parser::QubitListContext::qubit(size_t i) {
  return getRuleContext<OpenQASM3Parser::QubitContext>(i);
}

std::vector<tree::TerminalNode *> OpenQASM3Parser::QubitListContext::COMMA() {
  return getTokens(OpenQASM3Parser::COMMA);
}

tree::TerminalNode* OpenQASM3Parser::QubitListContext::COMMA(size_t i) {
  return getToken(OpenQASM3Parser::COMMA, i);
}


size_t OpenQASM3Parser::QubitListContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleQubitList;
}


std::any OpenQASM3Parser::QubitListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitQubitList(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::QubitListContext* OpenQASM3Parser::qubitList() {
  QubitListContext *_localctx = _tracker.createInstance<QubitListContext>(_ctx, getState());
  enterRule(_localctx, 26, OpenQASM3Parser::RuleQubitList);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(166);
    qubit();
    setState(171);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == OpenQASM3Parser::COMMA) {
      setState(167);
      match(OpenQASM3Parser::COMMA);
      setState(168);
      qubit();
      setState(173);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool OpenQASM3Parser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 7: return exprSempred(antlrcpp::downCast<ExprContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool OpenQASM3Parser::exprSempred(ExprContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 2);

  default:
    break;
  }
  return true;
}

void OpenQASM3Parser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  openqasm3ParserInitialize();
#else
  ::antlr4::internal::call_once(openqasm3ParserOnceFlag, openqasm3ParserInitialize);
#endif
}
