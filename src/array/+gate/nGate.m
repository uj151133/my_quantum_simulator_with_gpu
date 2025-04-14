function N = nGate(a, b, c)
%NGATE この関数の概要をここに記述
%   詳細説明をここに記述
N = [exp(1i * c) * cos(a - b), 0, 0, 1i * exp(1i * c) * sin(a - b); 0, exp(-1i * c) * cos(a + b), 1i * exp(-1i * c) * sin(a + b), 0; 0, 1i * exp(-1i * c) * sin(a + b), exp(-1i * c) * cos(a + b), 0; 1i * exp(1i * c) * sin(a - b), 0, 0, exp(1i * c) * cos(a - b)];
end

