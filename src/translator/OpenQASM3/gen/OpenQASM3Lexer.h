
// Generated from /home/ark/my_quantum_simulator_with_gpu/src/translator/OpenQASM3/OpenQASM3.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  OpenQASM3Lexer : public antlr4::Lexer {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, GPHASE = 10, ID = 11, P = 12, X = 13, Y = 14, Z = 15, 
    SX = 16, H = 17, S = 18, SDG = 19, T = 20, TDG = 21, RX = 22, RY = 23, 
    RZ = 24, CX = 25, CY = 26, CZ = 27, CP = 28, CRX = 29, CRY = 30, CRZ = 31, 
    CH = 32, CU = 33, SWAP = 34, CCX = 35, CSWAP = 36, U1 = 37, U2 = 38, 
    U3 = 39, RESET = 40, BARRIER = 41, MEASURE = 42, PI = 43, DECIMAL = 44, 
    NUMBER = 45, IDSTR = 46, STRING = 47, LBRACKET = 48, RBRACKET = 49, 
    LPAREN = 50, RPAREN = 51, COMMA = 52, SEMICOLON = 53, WS = 54
  };

  explicit OpenQASM3Lexer(antlr4::CharStream *input);

  ~OpenQASM3Lexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

