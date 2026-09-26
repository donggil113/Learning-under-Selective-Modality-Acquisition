# [연구 N1] When Does Modality Dropout Predict Deployment Risk? — Learning under Selective Modality Acquisition

**핵심 질문.** 완전 관측 집단에서 수행한 artificial modality dropout 평가는 언제 자연 결측을 가진 deployment population의 모델 순위를 설명하는가?

## 0. 판정 요약

| 구성 요소 | 판정 | 한 줄 근거 |
|---|---|---|
| 후보 방법 "acquisition-aware model selection" | **NO_GO** | 위험 **차**에 대한 결합 민감도 구간은 Guerdan et al. (ICML 2024), tilt는 Steingrimsson et al. (Biometrics 2024), Γ=1은 표준 IW/DR이다. 실제 자료(PhysioNet 2012)에서 같은 정보로 dropout+bootstrap 기준선보다 많은 쌍을 맞히지 못한다(§7.5) |
| 획득(A = 검사 시행) 수준의 주장 | **BLOCKED_REALDATA** | PhysioNet 2012·UCI Heart는 "기록 없음 = 미시행"을 문서로 확인할 수 없다. MIMIC-IV 전체·CXR·ECG·Note와 eICU 전체는 credentialing이 필요하다. 공개 MIMIC-IV demo에서 X선 order 91건이 "파일 없음"으로 보인다(§6.3) |
| 기록 가용성(M) 수준의 실증 감사 (C4) | 수행 완료. 결과는 대체로 **음성** | 완전사례 코호트의 인공 dropout은 자연 결측 순위를 τ 0.63–0.77로 재현하고, 유의한 반전은 run당 약 1쌍(진실-유의 쌍의 약 0.5%)이다. 반전은 정보성 결측을 쓰는 mask-aware 모델에 몰리며, 절반가량은 class balance, 교차 ICU에서는 대부분이 관측 공변량 보정(DR)으로 설명된다. label 의존 기록은 실재한다(4개 panel이 모두 기록되지 않은 stay의 OR 0.46 [0.36, 0.58]) |
| 반합성 검증 (실제 특징·label, 알려진 기전) | 이론과 일치 | MCAR/MAR에서는 모든 평가가 비슷하다. label 의존이 커지면 인공 dropout은 확신을 갖고 틀린다(δ=0.5에서 결정 정답률 68%, δ=1.0에서 38%). AA는 참 λ에서 CI 포함률 94–99%를 유지하지만 결정을 거의 하지 못한다(δ=1.0에서 0.1–1.3%) |
| 최소 반례·가정 사다리 | 정확히 증명·수치 검증 | CE1: 관측 가능한 모든 정보가 같은 두 target 세계에서 순위가 반대다. CE2: S만으로 순위가 뒤집히고 IW로 복구된다. 새로움은 marginal(Stokes App. A, Rockenschaub Thm 1의 따름정리) |

**수행 항목별 위치:** 1 → §1–2, 2 → §2 표, 3 → §3, 4 → §4, 5 → §5·§7.5, 6 → §7.1–7.2, 7 → §6, 8 → §7.2–7.4.
