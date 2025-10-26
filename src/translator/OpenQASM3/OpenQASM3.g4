grammar OpenQASM3;

program
  : 'OPENQASM' NUMBER '.' NUMBER SEMICOLON
    includeStmt*
    statement*
  ;

includeStmt
  : 'include' STRING SEMICOLON
  ;

statement
  : qregDecl
  | cregDecl
  | gateStmt
  | measureStmt
  | barrierStmt
  ;

qregDecl
  : 'qreg' IDSTR LBRACKET NUMBER RBRACKET SEMICOLON
  ;

cregDecl
  : 'creg' IDSTR LBRACKET NUMBER RBRACKET SEMICOLON
  ;

gateStmt
  : gateName (LPAREN paramList? RPAREN)? gateArgs SEMICOLON
  ;

paramList
  : expr (COMMA expr)*
  ;

expr
  : NUMBER
  | IDSTR
  | 'pi'
  | expr ('+' | '-' | '*' | '/') expr
  | LPAREN expr RPAREN
  ;

gateName
  : X | Y | Z | H | S | SDG | T | TDG
  | RX | RY | RZ
  | CX | CY | CZ | CP
  | SWAP | CCX | CSWAP
  | U1 | U2 | U3
  | SX | CH | CRX | CRY | CRZ | CU
  | P | ID | GPHASE |
  ;

gateArgs
  : qubit (COMMA qubit)*
  ;

qubit
  : IDSTR LBRACKET NUMBER RBRACKET
  ;

measureStmt
  : MEASURE qubit '->' IDSTR LBRACKET NUMBER RBRACKET SEMICOLON
  ;

barrierStmt
  : BARRIER qubitList SEMICOLON
  ;

qubitList
  : qubit (COMMA qubit)*
  ;

// ----------------- Lexer Rules -----------------
GPHASE : 'gphase' ;
ID     : 'id' ;
P      : 'p' | 'phase' ;
X      : 'x' ;
Y      : 'y' ;
Z      : 'z' ;
SX     : 'sx' ;
H      : 'h' ;
S      : 's' ;
SDG    : 'sdg' ;
T      : 't' ;
TDG    : 'tdg' ;
RX     : 'rx' ;
RY     : 'ry' ;
RZ     : 'rz' ;
CX     : 'cx' | 'CX' ;
CY     : 'cy' ;
CZ     : 'cz' ;
CP     : 'cp' | 'cphase' ;
CRX    : 'crx' ;
CRY    : 'cry' ;
CRZ    : 'crz' ;
CH     : 'ch' ;
CU     : 'cu' ;
SWAP   : 'swap' ;
CCX    : 'ccx' ;
CSWAP  : 'cswap' ;
U1     : 'u1' ;
U2     : 'u2' ;
U3     : 'u3' ;
RESET  : 'reset' ;
BARRIER: 'barrier' ;
MEASURE: 'measure' ;

NUMBER : [0-9]+ ;
IDSTR  : [a-zA-Z_][a-zA-Z_0-9]* ;
STRING : '"' ~["\r\n]* '"' ;
LBRACKET: '[' ;
RBRACKET: ']' ;
LPAREN: '(' ;
RPAREN: ')' ;
COMMA: ',' ;
SEMICOLON: ';' ;

WS : [ \t\r\n]+ -> skip ;
