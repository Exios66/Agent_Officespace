# Reasoning-augmented PokerBench SFT — worked examples

Eight hand-crafted PokerBench-style prompts, each shown in **two**
output formats:

- **Concise** — a 4–8 sentence chain-of-thought paragraph ending with
  a single `Decision: <action>` line. This is what
  `poker-predictor reason generate --style concise` (default) emits
  and is the recommended format for small (≤3B) student LLMs where
  context is precious.
- **Structured** — a three-section walkthrough (`### Strategic
  Analysis`, `### Mathematical Calculations`, `### Action`). Emitted
  by `poker-predictor reason generate --style structured`. More
  verbose, but each section can be graded independently during RL /
  DPO or verified programmatically.

Every example uses the same three-turn TRL chat layout:

```
[SYSTEM] <style-appropriate strategist / coach system prompt>
[USER]   <PokerBench-style situation prompt>
[ASSISTANT] <one of the two formats above>
```

Turn-by-turn JSONL versions of these same examples live at
[`reasoning_sft_examples.concise.jsonl`](reasoning_sft_examples.concise.jsonl)
and
[`reasoning_sft_examples.structured.jsonl`](reasoning_sft_examples.structured.jsonl).

---

## Example 1 — HU preflop, BB 3-bets AKo vs BTN min-raise

### [USER]

```
Format: Heads-Up NLH. Hero is BB holding [Ah, Ks]. Pot: 3BB. Effective Stacks: 99BB.
Board: (pre-flop)
Action: Pre-flop: BTN raises to 2BB, Hero to act.
```

### [ASSISTANT] — concise

```
Hero is BB with AhKs facing a BTN min-raise HU. AKo sits in the top ~5% of hands by equity against BTN's ~85% opening range and dominates most Ax and Kx that will continue. Flatting invites SPR problems on A- and K-high boards where BB's range is capped, so 3-betting resolves that asymmetry and takes initiative into the flop. A 4x IP-vs-OOP size (~8bb) also charges villain's dominated broadways enough to fold or play a bloated pot OOP.
Decision: raise 8.0bb
```

### [ASSISTANT] — structured

```
### Strategic Analysis
1. **Range vs Range**: BTN opens ~85% of hands HU; AKo is top-5% by equity and dominates Ax / Kx continues.
2. **Hand strength**: AKo is a premium value hand — it flips vs pairs and dominates most broadway hands that continue.
3. **Value vs bluff**: Pure value 3-bet; no need to balance from this frequency.
4. **GTO strategy**: 3-bet to build the pot with initiative rather than flat and play OOP on capped A / K high boards.

### Mathematical Calculations
* Pot size: 3 BB
* Villain's bet: 2 BB
* Target 3-bet: 8 BB (4x the open)
* Post-3bet pot if called: 17 BB, SPR ≈ 5.7

### Action
RAISE 8.0 BB
```

---

## Example 2 — 6-max preflop, SB opens QQ folded to

### [USER]

```
Format: 6-max NLH. Hero is SB holding [Qc, Qh]. Pot: 1.5BB. Effective Stacks: 100BB.
Board: (pre-flop)
Action: Pre-flop: UTG folds, HJ folds, CO folds, BTN folds, Hero to act.
```

### [ASSISTANT] — concise

```
Hero is SB with QQ folded around to a BB-vs-SB single-raised pot spot. QQ is a top-tier value hand and belongs firmly in SB's opening range — never limped, never mixed. Because BB will play back with a wide 3-bet range and cold-4bet only very rarely, SB prefers a slightly larger opening size to charge BB's flatting range and reduce the mixing burden postflop. Standard SB vs BB open is 3.0bb.
Decision: raise 3.0bb
```

### [ASSISTANT] — structured

```
### Strategic Analysis
1. **Range vs Range**: SB opens ~40% folded-through; QQ is at the very top of that range.
2. **Hand strength**: Third-nut pocket pair — pure value opener, never limped.
3. **Value vs bluff**: Pure value; QQ mixes 0% into slowplays here.
4. **GTO strategy**: Open larger (~3x) OOP to charge BB's wide flatting range and simplify OOP postflop.

### Mathematical Calculations
* Pot size: 1.5 BB (0.5 SB + 1.0 BB)
* Target open: 3.0 BB
* Post-open pot if BB flats: 6.5 BB, SPR ≈ 15

### Action
RAISE 3.0 BB
```

---

## Example 3 — 6-max preflop, BB flats 88 vs CO 2.5bb open

### [USER]

```
Format: 6-max NLH. Hero is BB holding [8h, 8d]. Pot: 4BB. Effective Stacks: 100BB.
Board: (pre-flop)
Action: Pre-flop: UTG folds, HJ folds, CO raises to 2.5BB, BTN folds, SB folds, Hero to act.
```

### [ASSISTANT] — concise

```
Hero is BB with 88 facing a CO open. Getting 1.5bb to call 1.5bb (3:1) with a pocket pair that plays reasonably well multiway and hits sets ~12% on the flop, calling clears easy. 3-betting is a mistake with 88 — CO folds out the hands 88 beats and continues with pairs and broadways that dominate 88 in a bloated pot. Flatting realises equity, keeps CO's range wide, and preserves set-mining EV at 100bb effective.
Decision: call
```

### [ASSISTANT] — structured

```
### Strategic Analysis
1. **Range vs Range**: CO opens ~27%; BB flats a merged calling range that includes small-mid pairs, suited broadways, and suited connectors.
2. **Hand strength**: 88 is a medium pair with excellent implied odds via set-mining.
3. **Value vs bluff**: 3-betting folds out worse and gets called by better — negative EV.
4. **GTO strategy**: Flat, realise equity, and stack CO's overpairs and top-pairs when hero flops a set.

### Mathematical Calculations
* Pot size: 4.0 BB (2.5 CO + 0.5 SB + 1.0 BB)
* Bet to call: 1.5 BB
* Pot odds: 1.5 / (4.0 + 1.5) = 27.3%
* Set-mining equity: ~12% flop-a-set × implied stack ≈ +EV @ 100bb
* SPR if called: ~19

### Action
CALL
```

---

## Example 4 — 6-max preflop, SB folds AQo facing CO 4-bet

### [USER]

```
Format: 6-max NLH. Hero is SB holding [Ah, Qc]. Pot: 22BB. Effective Stacks: 79BB.
Board: (pre-flop)
Action: Pre-flop: UTG folds, HJ folds, CO raises to 2.5BB, BTN folds, Hero 3-bets to 10BB, BB folds, CO 4-bets to 22BB.
```

### [ASSISTANT] — concise

```
Hero is SB with AQo facing a CO 4-bet after a squeeze from the SB. Against CO's 4-bet range (QQ+ / AK plus a very small bluff frequency) AQo has roughly 30% equity — enough to consider, but hero is OOP with SPR ~2.5 if called, and 5-bet jamming folds out the entire bluff frequency while getting stacks in against a range that dominates AQo 3:1. Calling is worse still because hero is capped, uncomfortable OOP, and has no clean flops. Folding preserves stack for a better spot.
Decision: fold
```

### [ASSISTANT] — structured

```
### Strategic Analysis
1. **Range vs Range**: CO's 4-bet range is ~QQ+/AK with a small suited-Ax bluff frequency; AQo has ~30% equity.
2. **Hand strength**: Dominated by the top of villain's value range (AK/AA/QQ), flips or is behind vs the rest.
3. **Value vs bluff**: 5-bet jam is -EV given villain's tight range; call is -EV OOP with SPR ~2.5.
4. **GTO strategy**: Fold; AQo is at the bottom of hero's 3-bet range and is a pure fold vs a 4-bet at these stack depths.

### Mathematical Calculations
* Pot size: 22.0 BB
* Bet to call: 12.0 BB
* Pot odds: 12 / (22 + 12) = 35.3%
* AQo equity vs CO 4-bet range: ~30% → insufficient to call
* SPR if called: 79 / 44 ≈ 1.8 (played OOP)

### Action
FOLD
```

---

## Example 5 — HU flop, BTN checks back Ax on wet 9♠8♠7♦

### [USER]

```
Format: Heads-Up NLH. Hero is BTN holding [Ad, Th]. Pot: 6BB. Effective Stacks: 97BB.
Board: [9s, 8s, 7d]
Action: Pre-flop: BTN raises to 3BB, BB calls. Flop: BB checks, Hero to act.
```

### [ASSISTANT] — concise

```
Hero is BTN with Ad-Th on 9s8s7d after opening and getting flatted. The board is one of the wettest possible textures in the game — every straight draw, all combo draws, and two flush-draws hit BB's flatting range hard, while BTN's high-card and overpair range gains only marginal equity. Ad-Th has just gutshot equity plus a backdoor nut-flush draw; c-betting bloats the pot OOP for BB to check-raise into. Solvers strongly prefer a high check frequency here — realise equity, let villain barrel with air, and reevaluate on the turn.
Decision: check
```

### [ASSISTANT] — structured

```
### Strategic Analysis
1. **Board texture**: 9s8s7d is monotone-adjacent — connects hard with BB's suited connectors, one-gappers, and small pairs.
2. **Hand strength**: Ace-high with a gutshot to J and a backdoor nut-flush draw — showdown value only.
3. **Value vs bluff**: Not enough equity to bet-fold; not enough fold-equity to bet-bluff on a texture that continues wide.
4. **GTO strategy**: Check back to realise equity, control pot size, and let BB do the bluffing on later streets.

### Mathematical Calculations
* Pot size: 6.0 BB
* Effective stack: 97 BB → SPR ~16
* Hand equity vs BB's flatting range: ~48% (mostly showdown value)
* Solver frequency for check: ~72% on this exact texture

### Action
CHECK
```

---

## Example 6 — 6-max flop, BB check-raises turned set on dry 9♣6♦2♠

### [USER]

```
Format: 6-max NLH. Hero is BB holding [6h, 6c]. Pot: 6BB. Effective Stacks: 97BB.
Board: [9c, 6d, 2s]
Action: Pre-flop: CO raises to 2.5BB, BTN folds, SB folds, Hero calls. Flop: Hero checks, CO bets 3BB, Hero to act.
```

### [ASSISTANT] — concise

```
Hero is BB with 66 on 9c6d2s after flatting a CO open — a middle set on the second-driest possible flop. CO's c-bet range is very wide (~65%) and mostly overpairs, TT-QQ overpairs, Ax high-card, and 9x top-pair, so 66 dominates a huge chunk of the continuing range. Because the board is dry there are almost no draws to protect against, and slowplaying keeps CO barreling turns with air. But BB's range is check-heavy here and CO barrels turns often enough that a check-raise both protects the check range and gets value from top pair / overpairs that will pay a 3-4x raise. A standard 3.5-4x c-bet raise is ideal.
Decision: raise 12.0bb
```

### [ASSISTANT] — structured

```
### Strategic Analysis
1. **Board texture**: 9c6d2s is rainbow, disconnected, and static — almost no draws to protect against.
2. **Hand strength**: Middle set — a top-of-range monster that dominates CO's overpairs and top-pair range.
3. **Value vs bluff**: Pure value raise; slowplaying costs value on later streets when 9x check-folds turns.
4. **GTO strategy**: Check-raise ~3.5-4x c-bet size to charge overpairs and top-pair while denying free realisation to backdoor equity.

### Mathematical Calculations
* Pot size: 6.0 BB (before c-bet)
* Villain c-bet: 3.0 BB → pot 9.0 BB, bet-to-call 3.0 BB
* Target check-raise: 12.0 BB (4x the c-bet)
* Post-raise pot if called: 27 BB, SPR ≈ 3.5
* Set equity vs CO c-bet range: ~86%

### Action
RAISE 12.0 BB
```

---

## Example 7 — HU turn, BB half-pot barrels missed flush draw

### [USER]

```
Format: Heads-Up NLH. Hero is BB holding [Jh, 9h]. Pot: 12BB. Effective Stacks: 90BB.
Board: [Kh, 7h, 3c, 2d]
Action: Pre-flop: BTN raises to 2BB, Hero calls. Flop: Hero checks, BTN bets 3BB, Hero calls. Turn: Hero to act.
```

### [ASSISTANT] — concise

```
Hero is BB with Jh9h on Kh-7h-3c-2d after flatting a BTN min-raise and check-calling a small flop c-bet — hero holds the third-nut flush draw plus a gutshot to T. BTN's range on the turn after a small flop c-bet and no turn barrel is capped: mostly Kx and small pairs, with the strongest hands (sets, AK) already having sized up on the flop or bet turn. Leading into BTN accomplishes two things: (1) it caps hero's own range less than checking (BB check-calls a lot), and (2) it forces BTN's weak Kx and pocket pairs into a tough spot with no room to bluff-catch turns and rivers. A half-pot lead ~6bb is standard, but on a two-tone board that just missed a scare card, BB solver leads larger (~12bb) to charge Kx and pull the trigger on hero's own flush + straight draws.
Decision: bet 12.0bb
```

### [ASSISTANT] — structured

```
### Strategic Analysis
1. **Board texture**: K-high two-tone with a blank turn — BTN's turn-checking range is capped.
2. **Hand strength**: Third-nut flush draw + gutshot = ~30% equity vs BTN's range; strong semi-bluff.
3. **Value vs bluff**: Semi-bluff — folds out most of BTN's small pairs and steals K-x's showdown value on rivers.
4. **GTO strategy**: Lead pot-sized (~12bb) to charge Kx, force pocket pairs to fold, and set up a river shove on flush completions.

### Mathematical Calculations
* Pot size: 12.0 BB
* Target lead: 12.0 BB (pot-sized)
* Post-lead pot if called: 36.0 BB, SPR ≈ 2.2
* Hand equity vs BTN's turn range: ~30%
* Fold equity required: (12 / (12 + 12)) = 50% — realistic vs a capped range.

### Action
BET 12.0 BB
```

---

## Example 8 — HU river, BB overbets 54s cracking Ax on A-K-8-3-2

### [USER]

```
Format: Heads-Up NLH. Hero is BB holding [5s, 4s]. Pot: 24BB. Effective Stacks: 76BB.
Board: [As, Kd, 8c, 3h, 2d]
Action: Pre-flop: BTN raises to 2BB, Hero calls. Flop: Hero checks, BTN bets 3BB, Hero calls. Turn: Hero checks, BTN checks. River: Hero is first to act.
```

### [ASSISTANT] — concise

```
Hero is BB with 5s4s on As-Kd-8c-3h-2d. The river 2d completes hero's back-door wheel (2-3-4-5-8-straight uses 5-4-3-2-A), giving hero a full straight while BTN's turn-check caps his range to weak Ax, weak Kx, and busted broadway backdoors — none of which beat a wheel. Because BTN's range is entirely bluff-catchers and hero blocks zero of them, a large-sized bet extracts maximum value from every A / K / pocket pair. Solvers strongly prefer a 75-100% pot bet on this river as BB, using nut hands like the wheel as the anchor of the value range.
Decision: bet 18.0bb
```

### [ASSISTANT] — structured

```
### Strategic Analysis
1. **Board texture**: The river [2d] completes a back-door straight for Hero (2-3-4-5-A wheel via 5-4).
2. **Hand strength**: Hero has successfully cracked the BTN's likely top-pair or middle-pair range — the nut straight.
3. **Value vs bluff**: This is an absolute value hand. BTN checked back the turn, capping their range to weak Aces, Kings, or missed draws. Hero must bet large to maximize value from bluff-catchers.
4. **GTO strategy**: Bet 75% to 100% of the pot to target BTN's Ax hands.

### Mathematical Calculations
* Pot Size: 24 BB
* Target Bet Size: 18 BB (75% pot)
* Post-bet pot if called: 60 BB (hero has 58 behind)
* Hand equity vs BTN's river range: ~100% (nut straight vs capped bluff-catchers)

### Action
BET 18.0 BB
```

---

## How to use these examples

1. **As a sanity check for a labeler run.** After a
   `poker-predictor reason generate` pass, diff a few of your
   labeler's outputs against these hand-authored references — the
   *shape* should match even if the wording differs.
2. **As a seed set for iterative distillation.** Start a small run
   with these 8 rows pre-loaded (concatenate the `.jsonl` file into
   your training set), then top up with a GPT-4o pass over the full
   PokerBench split.
3. **As a specification for a custom labeler.** If you're writing a
   PioSolver / GTO+ HTTP wrapper (see the contract in
   [`../../poker_predictor/llm/reasoning/README.md`](../../poker_predictor/llm/reasoning/README.md)),
   your wrapper's output should be reproducing the *structured* format
   for structured requests and the *concise* format for concise
   requests.
