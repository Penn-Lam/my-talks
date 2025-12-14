---
theme: seriph
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Lab 1: Feature Creation & Selection
  MCTS Variant Performance Prediction
drawings:
  persist: false
transition: slide-left
title: MCTS Variant Performance Prediction
mdc: true
---

# MCTS Variant Performance Prediction

Feature Engineering & Selection Analysis

<br>

<div class="text-2xl opacity-80">
林芃芃 | 郑炼鑫
</div>

<!--
大家好，我是林芃芃。今天代表小组分享我们的工作，我们在Kaggel上选了一道MCTS 变体性能预测的赛题，然后我们对其进行了特征工程。
-->

---
transition: fade-out
---

# Introduction

## Goal
Predict the relative performance of MCTS variants (Agent 1 vs Agent 2) based on algorithm configuration and game features.

Competition: [Game-Playing Strength of MCTS Variants](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants)

<v-click>

## Methodology
"Structure Analysis —> Statistical Aggregation —> Interaction Modeling"

</v-click>

<v-click>

1.  **Parse Components**: Extract Selection, Exploration, Playout, Score Bounds.
2.  **Aggregate Stats**: Use Out-of-Fold (OOF) target encoding for robust estimation.
3.  **Model Interactions**: Explore how Game Stochasticity regulates Agent differences.

</v-click>

<!--
这次比赛的目标很简单：在给定游戏环境下，预测 Agent 1 打 Agent 2 的胜率。

既然是预测“相对强弱”，我们没有把重点放在游戏本身的规则属性上，而是设计了一条“结构解析 —— 统计聚合 —— 交互建模”的特征工程路径。简单说，就是先拆解算法参数，用统计方法算强度分，最后看环境怎么影响这些强度的发挥。
-->

---

# Feature 1: ExplorationGap
<div class="text-gray-500 text-sm">Exploration Constant Difference</div>

<div class="grid grid-cols-[3fr_2fr] gap-4 mt-8">

<div>

### Definition
Measure the relative "exploration" tendency between two agents.

$$ \text{ExplorationGap} = \epsilon_1 - \epsilon_2 $$

<v-click>

### Key Insight
- **Positive Correlation**: $\rho = 0.107$ ($p \approx 0$)
- When Agent 1 is more "explorative", it generally achieves higher utility.
- **Extreme Case**: High vs Low exploration yields a stable mean utility advantage of <span class="text-red-400 font-bold">+0.131</span>.

</v-click>

</div>

<div class="flex flex-col gap-2 justify-center">
  <img src="/assets/ExplorationGap.png" class="h-65 object-contain rounded-lg shadow" />
</div>

</div>

<!--
来看第一个特征：探索常数差 (ExplorationGap)。也就是 Agent 1 和 Agent 2 谁更爱“探索”。我们发现一个很强的正相关规律：在这个数据集里，探索常数越大的代理，胜率通常越高。特别是当高探索打低探索时，均值优势能达到 +0.131。
-->

---

# Feature 1: ExplorationGap
<div class="text-gray-500 text-sm">Exploration Constant Difference</div>

<div class="flex flex-col gap-2 items-center justify-center h-full">
  <img src="/assets/feature1_robustness.png" class="h-75 object-contain rounded-lg shadow" />
  <div class="text-sm text-gray-500 mt-4">Robust across different Selection strategies (ProgressiveHistory, UCB1).</div>
</div>

<!--
而且这个规律非常稳健。大家看这张图，不管具体的 Selection 策略是用 UCB1 还是 ProgressiveHistory，这条红色的趋势线都是向上的。这说明“探索常数差”是一个普适性很强的主效应特征。
-->

---
layout: two-cols
---

# F1: Implementation

```python {all|2-3|5-7|all}
# 1. Construct Feature
df['ExplorationGap'] = (
    df['exploration1'] - df['exploration2']
)

# 2. Statistical Validation
rho, p = spearmanr(
    df['ExplorationGap'], 
    df['utility_agent1']
)

# 3. Bootstrap Analysis
# (See Appendix for full code)
```

::right::

<div class="ml-4 mt-16">

### Findings
- **Robustness**: Validated via Stratified Bootstrap ($N=2000$).
- **Conclusion**: A directionally clear, interpretable, and robust main effect feature.

<br>


> Higher exploration constant generally benefits Agent 1 in this dataset.

</div>

<!--
实现上，我们直接做差值构造，并通过了 Bootstrap 分层验证，置信区间很窄，信号很纯净。

第一步，直接计算两个 Agent 的参数差值，完成特征构造。

第二步，立刻计算它和 Utility 的 Spearman 相关系数，验证信号的显著性。

最后，大家看右边。 我们通过 2000 次的分层 Bootstrap 验证了它的稳健性。结论很清晰：探索常数越高，对 Agent 1 越有利。
-->

---

# Feature 2: Playout Strategy (Part 1)

<div class="grid grid-cols-2 gap-10 mt-8">


<div>

### Initial Hypothesis
Match vs Mismatch
- **Idea**: Do they use the same playout strategy?
- **Result**: <span class="text-red-400 font-bold">Failed</span>
  - $U \approx 5.94e9$, $p \approx 0.706$
  - "Same" vs "Different" has no significant impact.

</div>

<div>
<v-click>

### Refinement
Intensity Gap
Treat as **Ordered Pairs** (Agent 1 vs Agent 2):
- **MAST vs NST**: Highest Utility <span class="text-green-400 font-bold">(+0.118)</span>
- **NST vs MAST**: Lowest Utility <span class="text-red-400 font-bold">(-0.030)</span>

</v-click>
</div>

</div>

<!--
第二个特征关于模拟策略。

起初我们有个朴素的假设：如果双方策略不同，可能会有胜率偏差？结果数据打脸了，U 检验显示“策略是否相同”几乎没信息量。

于是我们转换视角，把对局看作有序组合。谁打谁很重要——比如 MAST 打 NST 优势巨大，反过来就很惨。
-->

---

# Feature 2: Playout Strategy (Part 2)

### PlayoutStrengthGap

$$ \Delta S_{playout} = S(P_1) - S(P_2) $$

<div class="flex flex-col gap-5 mt-6">

- **Correlation**: $\rho = 0.073$ ($p < 1e^{-200}$)
- A clear, monotonically increasing trend in utility.
- Although the effect is smaller than Selection Strategy, it is statistically significant.

<div class="flex justify-center w-full">
  <img src="/assets/feature2_strength_gap.png" class="rounded-lg shadow w-full object-contain max-h-40" />
</div>

</div>

<!--
基于此，我们计算了“模拟策略强度差”。

虽然它的相关性（0.073）比刚才的探索常数低一点，但看下面的一维趋势图，单调性非常完美。这证明了把 Categorical 变量转化为数值强度的思路是对的。
-->

---

# Feature 3: SelectionStrengthGap
<div class="text-sm text-gray-400">The Strongest Single Feature</div>

### Concept
Compute "Strength Score" for each (Selection + Exploration) configuration using **OOF Smoothing**.

$$ \text{SelectionStrengthGap} = \text{Strength}(A_1) - \text{Strength}(A_2) $$

<div class="flex justify-center gap-4 mt-6">
  <div class="w-1/2">
    <img src="/assets/feature3_analysis.png" class="rounded-lg shadow h-50 object-contain" />
  </div>
  <div class="w-1/2 flex flex-col justify-center">
    <ul class="space-y-4">
      <li v-click="1"><strong>Correlation:</strong> ρ = 0.220</li>
      <li v-click="2"><strong>Range:</strong> [-0.34, 0.34]</li>
      <li v-click="3"><strong>Monotonicity:</strong> Highly stable monotonic relationship with Utility.</li>
    </ul>
  </div>
</div>

<!--
接下来是全场最强的特征：选择策略强度差。

我们把 Selection 策略和探索常数结合，用 Out-of-Fold (OOF) 的方式算出了每种配置的“全局强度分”，然后做差。

结果非常漂亮：Spearman 相关系数达到 0.220，是所有单一特征里最高的，且单调性极好。
-->

---

# F3: Code Snippet

```python {monaco}
def oof_target_mean_smooth(df, col_name, target_col, n_splits=5, alpha=50):
    """
    Out-Of-Fold Target Encoding with Bayesian Smoothing
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    global_mean = df[target_col].mean()
    
    for train_idx, val_idx in kf.split(df):
        X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
        
        # Aggregation
        stats = X_train.groupby(col_name)[target_col].agg(['sum', 'count'])
        
        # Smoothing
        smoothed = (stats['sum'] + alpha * global_mean) / (stats['count'] + alpha)
        
        # Map to validation set
        oof[val_idx] = X_val[col_name].map(smoothed).fillna(global_mean)
        
    return oof
```

<!--
这是核心代码实现。关键点在于用了 KFold + 贝叶斯平滑，防止在计算强度分时发生数据泄露
-->

---

# Feature 4: Interaction Effect
SelectionStrengthGap $\times$ Game Stochasticity

Does luck (Stochasticity) dilute skill (Selection Strength)?

<div class="grid grid-cols-2 gap-8 mt-6">

<div>

### Findings
- **Deterministic**: $\rho \approx 0.210$
- **Stochastic**: $\rho \approx 0.317$ <span class="text-red-400 font-bold">(+51%)</span>
- **Result**: <span class="text-red-400 font-bold">Counter-intuitive!</span>
  
In stochastic games, the advantage of a stronger selection strategy is **amplified**, not diminished.

<div class="mt-4 p-4 bg-gray-800 rounded opacity-80 text-sm">
"Noise amplifies skill" — Stronger agents are more robust to randomness (or "noise") in the environment.
</div>

</div>

<div>
  <img src="/assets/feature4_analysis.png" class="rounded-lg shadow h-70 object-contain" />
</div>

</div>

<!--
第四个特征是我们最有意思的发现：交互效应。

我们想知道：游戏的随机性（运气成分）会不会稀释掉策略实力的作用？

直觉上，运气越重，实力越不重要对吧？
但数据结果完全反直觉！

在随机性高的游戏中，强策略的优势反而被放大了（相关性提升了 51%）。
这说明：越是嘈杂的环境，越需要强的算法来抗噪。
-->

---

# Negative Findings

We investigated features based purely on **Game Attributes**, but they showed weak signals.

| Feature | Description | Result |
| --- | --- | --- |
| **BoardSizeSweetSpot** | Is board area in "sweet spot" [2, 25]? | $\rho = -0.030$. Only local, weak lift. |
| **StabilityPace** | Game duration vs Turns variance. | $\rho \approx 0$. Long-tail noise. |
| **ProofFriendliness** | Interaction of ScoreBounds & Game provability. | $\rho \approx 0.005$. Mechanism valid but effect negligible. |

<br>

<v-click>


> **Takeaway**: Static game attributes are poor predictors on their own. They work best as **regulators** for Agent-based features.

</v-click>

<!--
为了严谨，我们也汇报一些失败的尝试。

我们试了棋盘大小、游戏节奏、可证明性等纯游戏属性特征。结果发现，它们单独对胜负的预测力几乎为零。

这再次印证了我们的观点：环境属性更多是作为“调节器”存在的，主导胜负的还是算法本身的差异。
-->

---
layout: center
class: text-center
---

# Conclusion

<div class="text-left max-w-2xl">

1.  **ExplorationGap** ($\rho=0.107$): More exploration $\rightarrow$ Higher utility.
2.  **PlayoutStrengthGap** ($\rho=0.073$): Modeling strategy Strength > Simple Matching.
3.  **SelectionStrengthGap** ($\rho=0.220$): Strongest main effect.
4.  **Interaction** ($\rho=0.215$): Stochasticity **amplifies** the effect of Selection Strength ($+51\%$).

<br>

**Final Thought**:
In complex game systems, **Agent Mechanism Differences** are the primary source of performance variance, while **Environment Characteristics** act as regulators.

</div>

<!--
最后总结一下：
- 探索常数和选择策略的差异提供了最强的主效应信号。
- 模拟策略不能只看异同，要算强度差。
- 最重要的是，环境随机性是一个放大器，它会放大强者对弱者的优势。


以上就是我们的分析，谢谢大家。
-->

---
layout: end
---

# Thank You!
