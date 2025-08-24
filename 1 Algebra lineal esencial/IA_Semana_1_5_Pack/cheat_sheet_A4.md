# Cheat-sheet A4 — Álgebra/Cálculo/Probabilidad para IA

## Álgebra
- Proyección: P = X (X^T X)^{-1} X^T (si X^T X es PD).
- SVD: X = U Σ V^T → columnas(V) = componentes; Σ^2/(n-1) = varianzas.
- tr(AB)=tr(BA); tr(A^T A)=||A||_F^2.
- Normas: ||x||_2, ||x||_1, ||x||_∞.
## Cálculo matricial
- d/dX tr(AX) = A^T
- d/dX ||AX-b||^2 = 2 A^T (AX-b)
- d/dX log det X = (X^{-1})^T (X PD)
## Optimización
- GD, Momentum, Adam; early stopping.
## Probabilidad
- E[aX+b]=aE[X]+b; Var(aX+b)=a^2 Var(X).
- KL, entropía, logística (gradiente X^T(σ(Xw)-y)/n).
## Series
- ACF/PACF; WAPE, sMAPE, MASE.
