
// Generated from OpenQASM3.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  OpenQASM3Lexer : public antlr4::Lexer {
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

