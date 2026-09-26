## 1. 저장소와 기존 결과 점검 (수행 1)

| 대상 | 상태 | N1에 주는 의미 |
|---|---|---|
| `donggil113/Learning-under-Selective-Modality-Acquisition` (본 저장소) | 원격·로컬 모두 **커밋 0개**(빈 저장소). 기존 코드·결과 없음 | 모든 결과를 이 세션에서 새로 생성. "기존 결과"로 인용할 것이 없음 |
| `donggil113/Identification-and-Sensitivity-under-Missingness-Shift` | 빈 저장소(커밋 0개) | 인용할 결과 없음 |
| `donggil113/Subset-Compatibility-in-Multimodal-Generation` | 빈 저장소(커밋 0개) | 인용할 결과 없음 |
| `donggil113/multimodal_paper` (INFOGAIN) | 코드·원고 존재. **모든 수치가 시뮬레이터 산출**, MIMIC-IV 추출은 "구현되었으나 실행되지 않음"(`README.md`, `STATUS.md`) | 실제 자료 결과 없음. 아래 세 가지 설계 문제를 N1에서 피해야 함 |

INFOGAIN에서 확인한, N1이 반복하면 안 되는 세 가지:

1. **S를 M으로 정의함.** `PREREGISTRATION.md` §1은 "±24시간 안에 필수 검사군이 모두 있는 ECG"만 포함한다. 즉 모집단 선택 S가 관측 mask M의 함수다. 이 코호트 안에서 수행한 어떤 인공 dropout 평가도 "완전 관측으로 선택된" 모집단에 대한 평가다.
2. **파일 부재를 미시행으로 코딩함.** `docs/DATA_ACCESS.md` §3: "A missing module is not an error: its modality is simply 'not acquired' for every patient". 이것은 수행 7이 금지한 바로 그 규칙이다(§6.3의 MIMIC-IV demo 감사 참고).
3. **MAR 가정이 검정 불가**하다고 스스로 밝히고(`STATUS.md` "Identification") MNAR 민감도 분석만 제시. 실제 자료에서는 한 번도 검증되지 않음.

## 2. 형식화: S, M, A의 분리 (수행 1–2)

단위 i마다 다음을 구분한다.

* **E** (적격 모집단): 적격 기준과 예측 시점 τ로만 정의. 어떤 변수의 존재 여부로도 정의하지 않는다.
* **S** (모집단 선택): 단위가 어느 모집단/코호트에 속하는가. 완전사례 설계에서는 개발 분할 안에서 `S=src ⇔ M=1`이므로 **S는 M의 함수**다. 교차 ICU 설계에서는 S가 ICU 유형이다.
* **M** (기록 가용성 mask): τ 시점에 모델이 읽을 수 있는 modality. 배포된 모델이 보는 것은 M이다.
* **A** (획득 = 검사 시행): 실제로 검사가 시행되었는가. **M과 같지 않다.** 시행되었으나 τ까지 결과가 저장되지 않았을 수 있고(결과 지연), 시행되었으나 추출 파일/모듈에 없을 수 있다(파일 부재). 따라서 이 보고서의 모든 실제 자료 주장은 M에 대한 것이며, A(임상의의 선택적 획득 결정)에 대한 주장은 §6.3의 이유로 **BLOCKED_REALDATA**다.

### 접근 가능한 정보 (수행 2)

| 당사자 | 입력 | mask | label | 비고 |
|---|---|---|---|---|
| D (개발) | X_obs | 자연 M | 있음 | 후보 모델 학습에만 사용 |
| V (완전 관측 검증 코호트) | 모든 panel | M ≡ 1 | 있음 | "완전 관측 집단". 완전사례로 **선택된** 모집단 |
| U (배포 표본, 선택 시점) | X_obs | 자연 M | **없음** | 선택 규칙이 읽을 수 있는 배포 정보 |
| T (배포 진실) | X_obs | 자연 M | 있음 | **평가 전용**. 어떤 선택 규칙도 읽지 않음 |
| `lab_n` 규칙 | U에서 n개 | 자연 M | n개만 공개 | 별도 정보 체제로만 보고 |

**비식별성 주장 범위.** PhysioNet 2012는 세 세트 모두 label이 공개되어 있다. 따라서 이 자료에서 목표 위험은 직접 추정 가능하며, 이 보고서는 이 자료에 대해 어떠한 비식별성도 주장하지 않는다. 비식별성(§4)은 "선택 시점에 target label이 없는" 정보 집합 {V, U}에 대한 진술일 뿐이며, 실제 자료에서는 T의 label로 그 가정 위반의 크기를 직접 측정한다(§7.4).

## 3. 선행연구 비교 (수행 3)

두 기준 논문은 모두 원문을 확인했다(Zhou et al.: arXiv 2211.02093 v3 전문, PMLR 206:9577–9606; Stokes et al.: arXiv 2504.00322 v1 전문).

| 항목 | Domain Adaptation under Missingness Shift (Zhou, Balakrishnan, Lipton, AISTATS 2023) | Domain Adaptation Under MNAR Missingness (Stokes, Do, Blecker, Chunara, Adhikari, arXiv 2025) |
|---|---|---|
| 불변 가정 | 완전자료 P(X,Y)가 두 도메인에서 동일(Def. 1). **S가 없음** | P(X_o,X_u) 및 P(Y∣X_o,X_u,R) 동일. S가 없음 |
| 결측 기전 | 주: UCAR(지시자 없음, 특징별 독립 Bernoulli). Prop. 1: 지시자 관측 + MCAR/v-MAR | MNAR(self-censoring + 잠재 교란) |
| source 접근 | label 있음, **source도 결측**(완전 관측 아님) | label 있음, source도 결측 |
| target 접근 | 비라벨 X̃_t | 비라벨 (X_o, X*_u, R) |
| 식별 결과 | 상대 결측률 식별(Rem. 5) → 라벨 target 분포 식별(Thm 5.2). **Alg. 1 = target 비율로 보정한 인공 dropout**. Prop. 1: 지시자+MCAR/v-MAR이면 (x̃, ξ)에 대한 공변량 이동 → IW | App. A: MNAR에서 P_T(Y∣x_obs,R) ≠ P_S(Y∣x_obs,R) → 관측자료 IW 불충분. Thm 1은 대입 오차 → 목표 MSE의 **환원**(식별 아님) |
| 비식별 결과 | 절대 결측률 비식별(Rem. 4) | self-censoring은 비모수적으로 비식별(인용) |
| 방법 | 재마스킹(Alg. 1), 선형 폐형식(Alg. 2) | 도메인별 MNAR 대입 후 도메인 분류기 IW |
| 실험 | 합성·반합성은 **인공 MCAR**. eICU 자연 결측에서는 **적응법이 비적응 모델에 패배**, 저자들이 원인으로 "선택 편향(매우 다른 코호트), label shift, 유병률 변화"를 지목 | 합성 self-censoring 약 12만 회. eICU 자연 결측, "평균 대입보다 크게 낫지 않음" |
| 불확실성 | 지표 bootstrap | seed 5개 |
| 모델 순위/위험 차 | 다루지 않음 | 다루지 않음 |
| S vs M | 분리하지 않음(실패 원인으로만 언급) | 분리하지 않음 |
| target label | 평가 전용 | 평가 + "leak" 대입기 + 개념 이동 진단에 사용 |

**둘을 먼저 비교한 결론.** Zhou et al.은 N1 질문의 **긍정 쪽 답을 이미 준다.** 완전자료 법칙이 불변이고 target mask가 MCAR(또는 관측 지시자 + v-MAR)이면, target 비율로 보정한 인공 dropout(Alg. 1)이 라벨 target 분포를 재현하므로 **어떤 모델의 순위든** 식별된다. Stokes et al.은 **부정 쪽의 핵심 재료를 이미 준다.** MNAR이면 관측자료 조건부가 이동한다. 둘 다 (i) 완전 관측이지만 S로 선택된 source, (ii) 두 고정 모델의 위험 차 부호, (iii) Y 의존 획득에 대한 민감도·부분식별, (iv) 같은 실제 코호트에서의 인공 대 자연 순위 비교를 다루지 않는다.

**N1 후보 기여별 선행연구 판정.** 문헌 조사 workflow(에이전트 6개 군집 조사, 종합, 12개 반박 에이전트)의 판정:

| 후보 | 이미 하는 선행연구 | 남는 새로움 |
|---|---|---|
| C1 S/M 분리와 dropout–배포 격차의 3항 분해 | 분해 기법: Zhang et al. (ICML 2023), DISDE (Cai, Namkoong, Yadlowsky 2023). S/M 개념 구분: Zamanian et al. (J Pers Med 2024) | marginal. 두 모집단 설계에 적용한 것뿐 |
| C2 최소 반례 + 가정 사다리 | 비수송성: Stokes App. A, Rockenschaub et al. (2024) Thm 1·Cor. 1. 그래프 비식별: Nabi et al. (2020), Mohan & Pearl (2021) | marginal. 순위 반전은 한 줄짜리 따름정리 |
| C3 위험 **차**에 대한 odds-ratio 민감도 경계, 유병률 LP, bootstrap, 붕괴 Γ* | 두 정책 성능 **차**의 결합 구간(MSM/Rosenbaum Γ, DR, bootstrap): Guerdan, Coston, Holstein, Wu (ICML 2024). 수송된 위험의 tilt 민감도: Steingrimsson, Robertson, Dahabreh (Biometrics 2024). 기준선 대비 minimax regret: Kallus & Zhou (NeurIPS 2018). 관측되지 않은 교란 아래 예측 알고리즘 성능 평가 경계: Rambachan, Coston, Kennedy. percentile bootstrap 민감도: Zhao, Small, Bhattacharya (JRSS-B 2019). 유병률 제약 LP: Dorn & Guo (sharp IPW 민감도) 유형. 반박 에이전트 2개 모두 "already done, high confidence" | **marginal.** 이진 Y·점별 손실에서 위험 차는 p_T에 선형이므로 날카로운 경계는 구간 끝점 평가이고, 유병률 제약을 더하면 연속 배낭 LP다. mask 패턴별 tilt라는 대입만 새롭다 |
| C4 실제 코호트에서 인공 대 자연 순위를 통제된 방식으로 비교 | 현상 자체는 표로 보임: DrFuse (AAAI 2024), MedFuse (MLHC 2022), CareBench (2026), Groenwold (2020) | **moderate.** 고정 적격 모집단·cutoff, 같은 모델, 쌍별 부호 검정과 CI, 그리고 표본 선택·유병률 조정을 먼저 적용한 통제 비교는 확인된 선행연구가 없음 |

미검증 인용과 반박 에이전트의 판정 원문은 `results/literature/literature_workflow.json`에 있다(예: Stokes et al.의 학회 게재 여부는 확인 불가, arXiv v1만 확인).

## 4. 최소 반례와 가정 사다리 (수행 4)

모든 수치는 `python -m src.n1.theory`가 유리수로 정확히 계산하고 검증한다(`results/theory/verification.json`, `all_ok: true`).

### 4.1 CE1: 비율을 맞춘 dropout + 불변 완전자료 법칙 + 비라벨 target으로도 순위가 정해지지 않음

이진 modality X 하나, 이진 Y, 항상 관측되는 공변량 없음. 완전자료 법칙은 source와 target이 동일하다: P(X=1,Y=1)=3/8, P(1,0)=1/8, P(0,1)=1/8, P(0,0)=3/8. source는 완전 관측이다. 두 모델은 X가 관측되면 똑같이 P(Y∣X)를 출력하고, X가 없으면 A는 1/2(= E_S[Y], dropout으로 학습한 모델이 배우는 값), B는 3/8을 출력한다.

| target 세계 | P_T(M=1∣X,Y) | 관측 가능한 target 법칙 P_T(M, X·M) | P_T(Y=1∣M=0) | R_T(A) (Brier) | R_T(B) | 순위 |
|---|---|---|---|---|---|---|
| W1 MCAR | 모두 1/2 | (1/4, 1/4, 1/2) | 1/2 | **7/32** | 29/128 | A 우위 |
| W2 label 의존 | (x,y)=(1,1):2/3, (1,0):0, (0,·):1/2 | **동일** (1/4, 1/4, 1/2) | 3/8 | 3/16 | **23/128** | B 우위 |

비율을 맞춘 dropout 평가는 두 세계에서 **같은 값**(A 7/32, B 29/128)을 주므로 어느 쪽 순위와도 일치할 수 없다. source 법칙, target의 관측 가능 법칙, target mask 비율이 모두 같으므로 {V, U} 정보로는 순위를 결정할 수 없다. 이것이 가장 작은 경우다. label 의존이 없으면 r(x,y)=r(x)이고 r(x)는 P_T(M=1,X=x)=r(x)P(x)로 식별되므로, 반례에는 X가 주어진 뒤에도 Y에 대한 의존이 반드시 필요하다.

### 4.2 CE2: S(완전사례 선택)만으로 순위가 뒤집히고, 표준 재가중으로 복구됨

X0∈{0,1}은 항상 관측된다. 배포 모집단에서 P(X0=1)=1/2, P(M=1∣X0)=(1/10, 9/10)이다(X0에 대한 MAR, label 의존 없음). 개발 코호트 = 완전사례이므로 P_S(X0=1)=9/10이다. 결측 시 fallback은 A=3/10, B=P_S(Y=1)=7/10이다.

* target 비율 dropout(완전사례에서): A 207/800, B **143/800** → B 우위
* 배포 진실: A **143/800**, B 207/800 → A 우위
* P_T(x0)/P_S(x0)로 재가중하고 P_T(M∣x0)로 mask하면: 정확히 진실과 같음

즉 MAR 아래에서도 **mask 비율을 맞추는 것만으로는 부족하다.** S가 M의 함수이기 때문이다. 다만 이 경우는 표준 IW로 해결된다(Zhou et al. Prop. 1과 같은 부류).

### 4.3 분해 항등식

이진 Y이고 손실이 label에 대해 아핀인 경우(Brier, log loss):

R_T(f) − R_D^q(f) = Σ_m (P_T(m) − q(m)) E_S[ℓ_m]  (패턴 가중)
 + Σ_m P_T(m) Σ_x (P_T(x∣m) − P_S(x)) ℓ̄_S(x)  (공변량/선택: U로 식별)
 + Σ_m P_T(m) E_T[(p_T(x,m) − p_S(x)) (ℓ1 − ℓ0)(x) ∣ M=m]  (label 의존: {V,U}로 비식별)

두 모델 A, B의 차이에 대해 마지막 항은 **E_T[(p_T − p_S) · g_AB]**가 된다. 여기서 g_AB = (ℓ_A1 − ℓ_A0) − (ℓ_B1 − ℓ_B0)이며, Brier에서는 2(f_B − f_A), log loss에서는 logit f_B − logit f_A다. 따라서 **두 모델이 불완전 패턴에서 비슷하게 예측하면 순위는 label 의존 획득에 둔감하다.** 반대로 결측 처리 전략끼리의 비교는 예측 차이가 바로 불완전 패턴에 몰려 있으므로 이 항에 가장 크게 노출된다. 분해는 무작위 유한 사례 300개에서 오차 4.4e-16 이내로 성립한다(`verification.json`).

### 4.4 가정 사다리: 어떤 가정을 더해야 비교할 수 있는가

| 단계 | 추가 가정 | 타당한 평가 규칙 | 근거/선행 |
|---|---|---|---|
| L0 | 없음 | 없음. 순위 비식별 (CE1) | Stokes App. A, Rockenschaub Thm 1 |
| L1 | S 없음(완전자료 법칙 불변) + target MCAR | 비율 맞춘 dropout (`freq`/`dams_rate`) | Zhou et al. Thm 5.2 / Alg. 1 |
| L2 | M ⟂ (X_mod, Y) ∣ X0, S는 X0를 통해서만 선택 | X0 IW + P_T(M∣X0) mask (`sel`) | Zhou Prop. 1, 표준 IW. CE2가 L1 규칙의 실패를 보임 |
| L3 | Y ⟂ M ∣ X_o(m) (패턴별), 양의 확률(positivity) | 패턴별 공변량 IW / DM / DR (`pat`,`dm`,`dr`) | 패턴별 공변량 이동. ICYM2I(역방향 IPW), Ulichney & Coston(DML) |
| L4 | odds ratio(p_T : p_S) ∈ [1/Γ, Γ] (또는 panel당 λ, 한쪽 방향, 유병률 제약) | 위험 **차**의 날카로운 구간. 0을 배제할 때만 결정 (`aa`) | Guerdan et al. 2024, Steingrimsson et al. 2024 (방법으로서 새롭지 않음) |
| L4′ | L4 + 유병률 π_T 알려짐 | 연속 배낭 LP 경계 | Dorn & Guo 유형 |
| L5 | 배포 label n개 | 직접 추정 (`lab_n`) | 가정 없음, 분산만 존재 |
| (참고) | refreshment 표본 / additive non-ignorable | 식별 가능 | Hirano, Imbens, Ridder, Rubin 2001 |

## 5. 후보 방법의 위치 (수행 5)

"acquisition-aware model selection"을 다음과 같이 구현했다(`src/n1/theory.py::diff_bounds`, `diff_bounds_prevalence`, `breakdown_gamma`; `src/n1/evaluators.py::aa_pair`).

1. Γ=1에서는 **표준 DR 추정량**이다. 패턴별 결과모형 p_S(x_o(m))(V에서 적합)과 U에서의 plug-in, V에서 교차적합한 IW 잔차 보정으로 구성된다. 이것은 새 방법이 아니다(일반적인 importance weighting / DR).
2. Γ>1에서는 불완전 패턴 단위마다 p_T ∈ OR-band(p_S, Γ)로 두고, **두 모델의 위험 차**에 대해 결합 구간을 계산한다. 차이가 p에 선형이므로 끝점 평가가 날카로운 경계다. 각 모델의 경계를 따로 구해 빼는 것보다 좁다(`tests/test_theory.py`). 추가 변형은 panel당 λ(Γ_i = λ^{#결측 panel}), 한쪽 방향("결측 단위가 같은 x_o에서 더 위험하지 않다"), 유병률 등식 제약(LP), 순위 붕괴 Γ*다.
3. 불확실성: V와 U의 bootstrap(점 규칙) 및 T의 bootstrap(진실 차이 CI).

선행연구 판정(§3)에 따라, 이 방법은 **Guerdan et al. 2024 + Steingrimsson et al. 2024의 mask 패턴별 대입**이다. 따라서 새 방법으로 주장하지 않는다. 아래 실험에서 이 방법은 "가정을 명시했을 때 무엇을 결정할 수 있는가"를 보여주는 **도구**로만 쓴다.
