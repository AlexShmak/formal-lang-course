grammar QueryLanguage;

prog: stmt* EOF;

stmt: bindStmt | addStmt | removeStmt | declareStmt;

declareStmt: 'let' VAR 'is' 'graph';

bindStmt: 'let' VAR '=' expr;

removeStmt:
    'remove' ('vertex' | 'edge' | 'vertices') expr 'from' VAR;

addStmt: 'add' ('vertex' | 'edge') expr 'to' VAR;

expr: NUM | CHAR | VAR | edgeExpr | setExpr | regexExpr | selectExpr;

setExpr: '[' expr (',' expr)* ']';

edgeExpr: '(' expr ',' expr ',' expr ')';

regexExpr: regexTerm ('|' regexTerm)*;
regexTerm: regexFactor (('.' | '&') regexFactor)*;
regexFactor: regexBase ('^' regexRange)*;
regexBase: CHAR | VAR | '(' regexExpr ')';

regexRange: '[' NUM '..' NUM? ']';

selectExpr:
    varFilter? varFilter? 'return' VAR (',' VAR)? 'where' VAR 'reachable' 'from' VAR 'in' VAR 'by'
        expr;

varFilter: 'for' VAR 'in' expr;

VAR: [a-z] [a-z0-9]*;
NUM: [0] | [1-9] [0-9]*;
CHAR: '"' [a-z] '"' | '\'' [a-z] '\'';

WS: [ \t\r\n]+ -> skip;
EOF_TOKEN: '<EOF>' -> skip;
