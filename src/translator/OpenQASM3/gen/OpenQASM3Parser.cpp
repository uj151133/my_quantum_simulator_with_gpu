
// Generated from /home/ark/my_quantum_simulator_with_gpu/src/translator/OpenQASM3/OpenQASM3.g4 by ANTLR 4.13.2


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
      "program", "version", "includeStmt", "statement", "qregDecl", "cregDecl", 
      "gateStmt", "paramList", "expr", "gateName", "gateArgs", "qubit", 
      "measureStmt", "barrierStmt", "qubitList"
    },
    std::vector<std::string>{
      "", "'OPENQASM'", "'include'", "'qreg'", "'creg'", "'*'", "'/'", "'+'", 
      "'-'", "'->'", "'gphase'", "'id'", "", "'x'", "'y'", "'z'", "'sx'", 
      "'h'", "'s'", "'sdg'", "'t'", "'tdg'", "'rx'", "'ry'", "'rz'", "'cx'", 
      "'cy'", "'cz'", "", "'crx'", "'cry'", "'crz'", "'ch'", "'cu'", "'swap'", 
      "'ccx'", "'cswap'", "'u1'", "'u2'", "'u3'", "'reset'", "'barrier'", 
      "'measure'", "'pi'", "", "", "", "", "'['", "']'", "'('", "')'", "','", 
      "';'"
    },
    std::vector<std::string>{
      "", "", "", "", "", "", "", "", "", "", "GPHASE", "ID", "P", "X", 
      "Y", "Z", "SX", "H", "S", "SDG", "T", "TDG", "RX", "RY", "RZ", "CX", 
      "CY", "CZ", "CP", "CRX", "CRY", "CRZ", "CH", "CU", "SWAP", "CCX", 
      "CSWAP", "U1", "U2", "U3", "RESET", "BARRIER", "MEASURE", "PI", "DECIMAL", 
      "NUMBER", "IDSTR", "STRING", "LBRACKET", "RBRACKET", "LPAREN", "RPAREN", 
      "COMMA", "SEMICOLON", "WS"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,54,150,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,7,
  	14,1,0,1,0,1,0,1,0,5,0,35,8,0,10,0,12,0,38,9,0,1,0,5,0,41,8,0,10,0,12,
  	0,44,9,0,1,1,1,1,1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1,3,3,3,57,8,3,1,4,1,
  	4,1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,6,1,6,1,6,3,6,76,
  	8,6,1,6,3,6,79,8,6,1,6,1,6,1,6,1,7,1,7,1,7,5,7,87,8,7,10,7,12,7,90,9,
  	7,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,3,8,101,8,8,1,8,1,8,1,8,1,8,1,8,
  	1,8,5,8,109,8,8,10,8,12,8,112,9,8,1,9,1,9,1,10,1,10,1,10,5,10,119,8,10,
  	10,10,12,10,122,9,10,1,11,1,11,1,11,1,11,1,11,1,12,1,12,1,12,1,12,1,12,
  	1,12,1,12,1,12,1,12,1,13,1,13,1,13,1,13,1,14,1,14,1,14,5,14,145,8,14,
  	10,14,12,14,148,9,14,1,14,0,1,16,15,0,2,4,6,8,10,12,14,16,18,20,22,24,
  	26,28,0,4,1,0,44,45,1,0,5,6,1,0,7,8,1,0,10,39,151,0,30,1,0,0,0,2,45,1,
  	0,0,0,4,47,1,0,0,0,6,56,1,0,0,0,8,58,1,0,0,0,10,65,1,0,0,0,12,72,1,0,
  	0,0,14,83,1,0,0,0,16,100,1,0,0,0,18,113,1,0,0,0,20,115,1,0,0,0,22,123,
  	1,0,0,0,24,128,1,0,0,0,26,137,1,0,0,0,28,141,1,0,0,0,30,31,5,1,0,0,31,
  	32,3,2,1,0,32,36,5,53,0,0,33,35,3,4,2,0,34,33,1,0,0,0,35,38,1,0,0,0,36,
  	34,1,0,0,0,36,37,1,0,0,0,37,42,1,0,0,0,38,36,1,0,0,0,39,41,3,6,3,0,40,
  	39,1,0,0,0,41,44,1,0,0,0,42,40,1,0,0,0,42,43,1,0,0,0,43,1,1,0,0,0,44,
  	42,1,0,0,0,45,46,7,0,0,0,46,3,1,0,0,0,47,48,5,2,0,0,48,49,5,47,0,0,49,
  	50,5,53,0,0,50,5,1,0,0,0,51,57,3,8,4,0,52,57,3,10,5,0,53,57,3,12,6,0,
  	54,57,3,24,12,0,55,57,3,26,13,0,56,51,1,0,0,0,56,52,1,0,0,0,56,53,1,0,
  	0,0,56,54,1,0,0,0,56,55,1,0,0,0,57,7,1,0,0,0,58,59,5,3,0,0,59,60,5,46,
  	0,0,60,61,5,48,0,0,61,62,5,45,0,0,62,63,5,49,0,0,63,64,5,53,0,0,64,9,
  	1,0,0,0,65,66,5,4,0,0,66,67,5,46,0,0,67,68,5,48,0,0,68,69,5,45,0,0,69,
  	70,5,49,0,0,70,71,5,53,0,0,71,11,1,0,0,0,72,78,3,18,9,0,73,75,5,50,0,
  	0,74,76,3,14,7,0,75,74,1,0,0,0,75,76,1,0,0,0,76,77,1,0,0,0,77,79,5,51,
  	0,0,78,73,1,0,0,0,78,79,1,0,0,0,79,80,1,0,0,0,80,81,3,20,10,0,81,82,5,
  	53,0,0,82,13,1,0,0,0,83,88,3,16,8,0,84,85,5,52,0,0,85,87,3,16,8,0,86,
  	84,1,0,0,0,87,90,1,0,0,0,88,86,1,0,0,0,88,89,1,0,0,0,89,15,1,0,0,0,90,
  	88,1,0,0,0,91,92,6,8,-1,0,92,93,5,50,0,0,93,94,3,16,8,0,94,95,5,51,0,
  	0,95,101,1,0,0,0,96,101,5,44,0,0,97,101,5,45,0,0,98,101,5,43,0,0,99,101,
  	5,46,0,0,100,91,1,0,0,0,100,96,1,0,0,0,100,97,1,0,0,0,100,98,1,0,0,0,
  	100,99,1,0,0,0,101,110,1,0,0,0,102,103,10,7,0,0,103,104,7,1,0,0,104,109,
  	3,16,8,8,105,106,10,6,0,0,106,107,7,2,0,0,107,109,3,16,8,7,108,102,1,
  	0,0,0,108,105,1,0,0,0,109,112,1,0,0,0,110,108,1,0,0,0,110,111,1,0,0,0,
  	111,17,1,0,0,0,112,110,1,0,0,0,113,114,7,3,0,0,114,19,1,0,0,0,115,120,
  	3,22,11,0,116,117,5,52,0,0,117,119,3,22,11,0,118,116,1,0,0,0,119,122,
  	1,0,0,0,120,118,1,0,0,0,120,121,1,0,0,0,121,21,1,0,0,0,122,120,1,0,0,
  	0,123,124,5,46,0,0,124,125,5,48,0,0,125,126,5,45,0,0,126,127,5,49,0,0,
  	127,23,1,0,0,0,128,129,5,42,0,0,129,130,3,22,11,0,130,131,5,9,0,0,131,
  	132,5,46,0,0,132,133,5,48,0,0,133,134,5,45,0,0,134,135,5,49,0,0,135,136,
  	5,53,0,0,136,25,1,0,0,0,137,138,5,41,0,0,138,139,3,28,14,0,139,140,5,
  	53,0,0,140,27,1,0,0,0,141,146,3,22,11,0,142,143,5,52,0,0,143,145,3,22,
  	11,0,144,142,1,0,0,0,145,148,1,0,0,0,146,144,1,0,0,0,146,147,1,0,0,0,
  	147,29,1,0,0,0,148,146,1,0,0,0,11,36,42,56,75,78,88,100,108,110,120,146
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

OpenQASM3Parser::VersionContext* OpenQASM3Parser::ProgramContext::version() {
  return getRuleContext<OpenQASM3Parser::VersionContext>(0);
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
    setState(30);
    match(OpenQASM3Parser::T__0);
    setState(31);
    version();
    setState(32);
    match(OpenQASM3Parser::SEMICOLON);
    setState(36);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == OpenQASM3Parser::T__1) {
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
      ((1ULL << _la) & 7696581393432) != 0)) {
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

//----------------- VersionContext ------------------------------------------------------------------

OpenQASM3Parser::VersionContext::VersionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* OpenQASM3Parser::VersionContext::DECIMAL() {
  return getToken(OpenQASM3Parser::DECIMAL, 0);
}

tree::TerminalNode* OpenQASM3Parser::VersionContext::NUMBER() {
  return getToken(OpenQASM3Parser::NUMBER, 0);
}


size_t OpenQASM3Parser::VersionContext::getRuleIndex() const {
  return OpenQASM3Parser::RuleVersion;
}


std::any OpenQASM3Parser::VersionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<OpenQASM3Visitor*>(visitor))
    return parserVisitor->visitVersion(this);
  else
    return visitor->visitChildren(this);
}

OpenQASM3Parser::VersionContext* OpenQASM3Parser::version() {
  VersionContext *_localctx = _tracker.createInstance<VersionContext>(_ctx, getState());
  enterRule(_localctx, 2, OpenQASM3Parser::RuleVersion);
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
    setState(45);
    _la = _input->LA(1);
    if (!(_la == OpenQASM3Parser::DECIMAL

    || _la == OpenQASM3Parser::NUMBER)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
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
  enterRule(_localctx, 4, OpenQASM3Parser::RuleIncludeStmt);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(47);
    match(OpenQASM3Parser::T__1);
    setState(48);
    match(OpenQASM3Parser::STRING);
    setState(49);
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
  enterRule(_localctx, 6, OpenQASM3Parser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(56);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case OpenQASM3Parser::T__2: {
        enterOuterAlt(_localctx, 1);
        setState(51);
        qregDecl();
        break;
      }

      case OpenQASM3Parser::T__3: {
        enterOuterAlt(_localctx, 2);
        setState(52);
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
      case OpenQASM3Parser::U3: {
        enterOuterAlt(_localctx, 3);
        setState(53);
        gateStmt();
        break;
      }

      case OpenQASM3Parser::MEASURE: {
        enterOuterAlt(_localctx, 4);
        setState(54);
        measureStmt();
        break;
      }

      case OpenQASM3Parser::BARRIER: {
        enterOuterAlt(_localctx, 5);
        setState(55);
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
  enterRule(_localctx, 8, OpenQASM3Parser::RuleQregDecl);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(58);
    match(OpenQASM3Parser::T__2);
    setState(59);
    match(OpenQASM3Parser::IDSTR);
    setState(60);
    match(OpenQASM3Parser::LBRACKET);
    setState(61);
    match(OpenQASM3Parser::NUMBER);
    setState(62);
    match(OpenQASM3Parser::RBRACKET);
    setState(63);
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
  enterRule(_localctx, 10, OpenQASM3Parser::RuleCregDecl);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(65);
    match(OpenQASM3Parser::T__3);
    setState(66);
    match(OpenQASM3Parser::IDSTR);
    setState(67);
    match(OpenQASM3Parser::LBRACKET);
    setState(68);
    match(OpenQASM3Parser::NUMBER);
    setState(69);
    match(OpenQASM3Parser::RBRACKET);
    setState(70);
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
  enterRule(_localctx, 12, OpenQASM3Parser::RuleGateStmt);
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
    setState(72);
    gateName();
    setState(78);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == OpenQASM3Parser::LPAREN) {
      setState(73);
      match(OpenQASM3Parser::LPAREN);
      setState(75);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 1257841302175744) != 0)) {
        setState(74);
        paramList();
      }
      setState(77);
      match(OpenQASM3Parser::RPAREN);
    }
    setState(80);
    gateArgs();
    setState(81);
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
  enterRule(_localctx, 14, OpenQASM3Parser::RuleParamList);
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
    setState(83);
    expr(0);
    setState(88);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == OpenQASM3Parser::COMMA) {
      setState(84);
      match(OpenQASM3Parser::COMMA);
      setState(85);
      expr(0);
      setState(90);
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

tree::TerminalNode* OpenQASM3Parser::ExprContext::DECIMAL() {
  return getToken(OpenQASM3Parser::DECIMAL, 0);
}

tree::TerminalNode* OpenQASM3Parser::ExprContext::NUMBER() {
  return getToken(OpenQASM3Parser::NUMBER, 0);
}

tree::TerminalNode* OpenQASM3Parser::ExprContext::PI() {
  return getToken(OpenQASM3Parser::PI, 0);
}

tree::TerminalNode* OpenQASM3Parser::ExprContext::IDSTR() {
  return getToken(OpenQASM3Parser::IDSTR, 0);
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
  size_t startState = 16;
  enterRecursionRule(_localctx, 16, OpenQASM3Parser::RuleExpr, precedence);

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
    setState(100);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case OpenQASM3Parser::LPAREN: {
        setState(92);
        match(OpenQASM3Parser::LPAREN);
        setState(93);
        expr(0);
        setState(94);
        match(OpenQASM3Parser::RPAREN);
        break;
      }

      case OpenQASM3Parser::DECIMAL: {
        setState(96);
        match(OpenQASM3Parser::DECIMAL);
        break;
      }

      case OpenQASM3Parser::NUMBER: {
        setState(97);
        match(OpenQASM3Parser::NUMBER);
        break;
      }

      case OpenQASM3Parser::PI: {
        setState(98);
        match(OpenQASM3Parser::PI);
        break;
      }

      case OpenQASM3Parser::IDSTR: {
        setState(99);
        match(OpenQASM3Parser::IDSTR);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(110);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(108);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(102);

          if (!(precpred(_ctx, 7))) throw FailedPredicateException(this, "precpred(_ctx, 7)");
          setState(103);
          _la = _input->LA(1);
          if (!(_la == OpenQASM3Parser::T__4

          || _la == OpenQASM3Parser::T__5)) {
          _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(104);
          expr(8);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(105);

          if (!(precpred(_ctx, 6))) throw FailedPredicateException(this, "precpred(_ctx, 6)");
          setState(106);
          _la = _input->LA(1);
          if (!(_la == OpenQASM3Parser::T__6

          || _la == OpenQASM3Parser::T__7)) {
          _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(107);
          expr(7);
          break;
        }

        default:
          break;
        } 
      }
      setState(112);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx);
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
  enterRule(_localctx, 18, OpenQASM3Parser::RuleGateName);
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
    setState(113);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 1099511626752) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
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
  enterRule(_localctx, 20, OpenQASM3Parser::RuleGateArgs);
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
    setState(115);
    qubit();
    setState(120);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == OpenQASM3Parser::COMMA) {
      setState(116);
      match(OpenQASM3Parser::COMMA);
      setState(117);
      qubit();
      setState(122);
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
  enterRule(_localctx, 22, OpenQASM3Parser::RuleQubit);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(123);
    match(OpenQASM3Parser::IDSTR);
    setState(124);
    match(OpenQASM3Parser::LBRACKET);
    setState(125);
    match(OpenQASM3Parser::NUMBER);
    setState(126);
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
  enterRule(_localctx, 24, OpenQASM3Parser::RuleMeasureStmt);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(128);
    match(OpenQASM3Parser::MEASURE);
    setState(129);
    qubit();
    setState(130);
    match(OpenQASM3Parser::T__8);
    setState(131);
    match(OpenQASM3Parser::IDSTR);
    setState(132);
    match(OpenQASM3Parser::LBRACKET);
    setState(133);
    match(OpenQASM3Parser::NUMBER);
    setState(134);
    match(OpenQASM3Parser::RBRACKET);
    setState(135);
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
  enterRule(_localctx, 26, OpenQASM3Parser::RuleBarrierStmt);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(137);
    match(OpenQASM3Parser::BARRIER);
    setState(138);
    qubitList();
    setState(139);
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
  enterRule(_localctx, 28, OpenQASM3Parser::RuleQubitList);
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
    setState(141);
    qubit();
    setState(146);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == OpenQASM3Parser::COMMA) {
      setState(142);
      match(OpenQASM3Parser::COMMA);
      setState(143);
      qubit();
      setState(148);
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
    case 8: return exprSempred(antlrcpp::downCast<ExprContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool OpenQASM3Parser::exprSempred(ExprContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 7);
    case 1: return precpred(_ctx, 6);

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
