grammar qasm;

mainprog: (header | qregDecl | cregDecl | iGate | xGate | zGate | sGate | sdgGate | hGate | tGate | tdgGate | swapGate | cxGate | czGate | rxGate | ryGate | rzGate | u3Gate | measureStmt)* EOF;

header: OPENQASM | INCLUDE;

qregDecl: 'qreg' ID '[' INT ']' ';' ;
cregDecl: 'creg' ID '[' INT ']' ';' ;

iGate: 'id' ID '[' INT ']' ';' ;

xGate: 'x' ID '[' INT ']' ';' ;
zGate: 'z' ID '[' INT ']' ';' ;
sGate: 's' ID '[' INT ']' ';' ;
sdgGate: 'sdg' ID '[' INT ']' ';' ;
hGate: 'h' ID '[' INT ']' ';' ;
tGate: 't' ID '[' INT ']' ';' ;
tdgGate: 'tdg' ID '[' INT ']' ';' ;

swapGate: 'swap' ID '[' INT ']' ',' ID '[' INT ']' ';' ;

cxGate: 'cx' ID '[' INT ']' ',' ID '[' INT ']' ';' ;
czGate: 'cz' ID '[' INT ']' ',' ID '[' INT ']' ';' ;

rxGate: 'rx' '(' expr ')' ID '[' INT ']' ';' ;
ryGate: 'ry' '(' expr ')' ID '[' INT ']' ';' ;
rzGate: 'rz' '(' expr ')' ID '[' INT ']' ';' ;

u3Gate: 'u3' '(' expr ',' expr ',' expr ')' ID '[' INT ']' ';' ;

measureStmt: 'measure' ID '[' INT ']' '->' ID '[' INT ']' ';' ;

OPENQASM : 'OPENQASM' WS+ '2.0' ';' ;
INCLUDE  : 'include' WS+ '"' .*? '"' ';' ;

ID  : [a-zA-Z_] [a-zA-Z0-9_]* ;
INT : [0-9]+ ;
PI: 'pi';
expr: exprMul (('*' | '/') exprMul)* ;
exprMul: sign? (PI | FLOAT | INT) ;
sign: ('+' | '-') ;
FLOAT : INT '.' INT ;
WS  : [ \t\r\n]+ -> skip ;

LINE_COMMENT : '//' ~[\r\n]* -> skip ;