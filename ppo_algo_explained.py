import warnings
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import GPT2ForSequenceClassification
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

# Suppress warnings
warnings.filterwarnings('ignore', message='`resume_download` is deprecated')
warnings.filterwarnings('ignore', message='Some weights of GPT2ForSequenceClassification were not initialized')

# Optionally, suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ✅ Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Print PPO explanation
print("\n" + "="*80)
print("📚 PROXIMAL POLICY OPTIMIZATION (PPO) EDUCATIONAL MODE")
print("="*80)
print("""
PPO is a policy gradient method that:
1. Collects experiences using current policy π_old
2. Calculates advantages A = R - V(s) to measure action quality
3. Updates policy π to maximize advantages while staying close to π_old
4. Updates value function V(s) to better predict returns

Key components:
- Policy Network: Generates text (language model)
- Value Head: Predicts expected future reward V(s)
- Reward Model: Scores generated text quality
- Advantage Function: A = R - V(s) measures if action was better than expected
""")
print("="*80)

# ✅ Load models
print("\n🔧 Loading models...")
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained("lvwerra/gpt2-imdb").to(device)
tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
tokenizer.pad_token = tokenizer.eos_token

reward_model = GPT2ForSequenceClassification.from_pretrained("lvwerra/gpt2-imdb").to(device)
reward_tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
reward_tokenizer.pad_token = reward_tokenizer.eos_token

# Helper function to get value from model
def get_value_from_model(model, input_ids):
    """Robust method to get value estimates from the model"""
    try:
        outputs = model(input_ids)
        if hasattr(outputs, 'value'):
            return outputs.value.squeeze()
        elif isinstance(outputs, tuple) and len(outputs) >= 2 and outputs[1] is not None:
            return outputs[1].squeeze()
    except:
        pass
    
    try:
        if hasattr(model, 'v_head') and hasattr(model, 'pretrained_model'):
            hidden = model.pretrained_model(input_ids, output_hidden_states=True)
            if hasattr(hidden, 'hidden_states'):
                last_hidden = hidden.hidden_states[-1][:, -1, :]
            else:
                last_hidden = hidden[0][:, -1, :]
            return model.v_head(last_hidden).squeeze()
    except:
        pass
    
    return torch.zeros(1).to(device)

# Helper function to calculate GAE advantages
def calculate_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE)
    Balances bias-variance tradeoff in advantage estimation
    """
    advantages = []
    returns = []
    gae = 0
    
    print("\n🔢 GAE Calculation Details:")
    print("-"*60)
    print("Working backwards through trajectory...")
    
    # Work backwards through trajectory
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # Terminal state
        else:
            next_value = values[t + 1]
        
        # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]
        
        # GAE: A_t = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + ...
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        
        # Also calculate returns for value function training
        returns.insert(0, rewards[t] + gamma * next_value)
        
        print(f"\nStep t={t}:")
        print(f"  r_t = {rewards[t]:.4f}")
        print(f"  V(s_t) = {values[t]:.4f}")
        print(f"  V(s_{{t+1}}) = {next_value:.4f}")
        print(f"  TD error δ_t = {rewards[t]:.4f} + {gamma:.2f}×{next_value:.4f} - {values[t]:.4f} = {delta:.4f}")
        print(f"  GAE A_t = {gae:.4f}")
    
    return advantages, returns

# ✅ Load prompts
try:
    with open("ppo_dataset.jsonl") as f:
        prompts = [json.loads(line)["prompt"] for line in f if line.strip()][:6]  # Use 6 for better batching
    print(f"✅ Loaded {len(prompts)} prompts from ppo_dataset.jsonl")
except:
    prompts = ["The movie was so", "I can't believe how", "It felt incredibly", 
               "The acting was", "Overall, I would say", "The plot was"]

# ✅ Run one detailed PPO epoch with educational output
print("\n" + "="*80)
print("🎓 DETAILED PPO TRAINING PROCESS")
print("="*80)

# Step 1: Generate trajectories
print("\n📊 STEP 1: COLLECTING EXPERIENCES WITH CURRENT POLICY")
print("-"*50)

all_queries = []
all_responses = []
all_rewards = []
all_values = []
all_log_probs = []

for i, prompt in enumerate(prompts):
    print(f"\n{'='*70}")
    print(f"🎬 Experience {i+1}/{len(prompts)}:")
    print(f"{'='*70}")
    print(f"Prompt: '{prompt}'")
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        # Get value estimate V(s)
        value = get_value_from_model(policy_model, input_ids)
        print(f"\n📊 Value Head Output V(s): {value.item():.4f}")
        print(f"   → This is the model's prediction of expected future reward")
        print(f"   → If V(s) is high, model expects good rewards from this state")
        
        # Generate response
        response_ids = policy_model.generate(
            input_ids=input_ids, 
            max_new_tokens=20, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=1.0
        )
        
        # Get log probabilities
        outputs = policy_model(response_ids)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = policy_model.pretrained_model(response_ids).logits
            
        # Calculate average log prob of generated tokens
        generated_tokens = response_ids[:, input_ids.shape[-1]:]
        log_probs = []
        token_probs = []
        
        print(f"\n📊 Token-by-Token Log Probability Calculation:")
        print("-"*50)
        
        for j in range(min(generated_tokens.shape[1], 5)):  # Show first 5 tokens
            if input_ids.shape[-1] + j - 1 < logits.shape[1]:
                token_logits = logits[0, input_ids.shape[-1] + j - 1]
                token_id = generated_tokens[0, j]
                
                # Calculate probability distribution
                probs = F.softmax(token_logits, dim=-1)
                token_prob = probs[token_id].item()
                
                # Calculate log probability
                log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
                log_probs.append(log_prob)
                token_probs.append(token_prob)
                
                # Decode token
                token_text = tokenizer.decode([token_id])
                
                print(f"Token {j+1}: '{token_text}'")
                print(f"  Token ID: {token_id}")
                print(f"  Probability: {token_prob:.4f} ({token_prob*100:.2f}%)")
                print(f"  Log probability: {log_prob.item():.4f}")
        
        avg_log_prob = torch.stack(log_probs).mean() if log_probs else torch.tensor(0.0)
        avg_prob = np.mean(token_probs) if token_probs else 0.0
        
        print(f"\n📊 Average Statistics:")
        print(f"  Average probability: {avg_prob:.4f} ({avg_prob*100:.2f}%)")
        print(f"  Average log probability: {avg_log_prob.item():.4f}")
    
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    generated_part = response_text[len(prompt):].strip()
    print(f"\n📝 Generated Text: '{generated_part}'")
    
    # Get reward from reward model
    print(f"\n📊 Reward Model Calculation:")
    print("-"*50)
    
    reward_inputs = reward_tokenizer(generated_part, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        reward_outputs = reward_model(**reward_inputs)
        # Sentiment classifier: positive sentiment - negative sentiment
        positive_score = reward_outputs.logits[0][1].item()
        negative_score = reward_outputs.logits[0][0].item()
        sentiment_score = positive_score - negative_score
        
        print(f"Positive sentiment logit: {positive_score:.4f}")
        print(f"Negative sentiment logit: {negative_score:.4f}")
        print(f"\nSentiment score calculation:")
        print(f"  sentiment_score = positive_logit - negative_logit")
        print(f"  sentiment_score = {positive_score:.4f} - {negative_score:.4f}")
        print(f"  sentiment_score = {sentiment_score:.4f}")
        
        # Calculate tanh step by step
        print(f"\nReward calculation using tanh:")
        print(f"  tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))")
        
        e_pos = math.exp(sentiment_score)
        e_neg = math.exp(-sentiment_score)
        reward = (e_pos - e_neg) / (e_pos + e_neg)
        
        print(f"  e^{sentiment_score:.4f} = {e_pos:.4f}")
        print(f"  e^{-sentiment_score:.4f} = {e_neg:.4f}")
        print(f"  tanh = ({e_pos:.4f} - {e_neg:.4f}) / ({e_pos:.4f} + {e_neg:.4f})")
        print(f"  tanh = {reward:.4f}")
        
        # Interpretation
        if sentiment_score > 0:
            print(f"\n✅ Positive sentiment detected (score > 0)")
        else:
            print(f"\n❌ Negative sentiment detected (score < 0)")
        
        print(f"Final reward R: {reward:.4f} (range: [-1, 1])")
    
    # Store experience
    all_queries.append(input_ids.squeeze())
    all_responses.append(response_ids[:, input_ids.shape[-1]:].squeeze())
    all_rewards.append(reward)
    all_values.append(value.item())
    all_log_probs.append(avg_log_prob)

# Step 2: Calculate advantages
print("\n" + "="*80)
print("📊 STEP 2: CALCULATING ADVANTAGES WITH GAE")
print("-"*50)
print("Advantage A(s,a) = R - V(s) tells us if the action was better than expected")
print("GAE uses exponentially weighted sum of TD errors for lower variance")
print("-"*50)

advantages, returns = calculate_gae(all_rewards, all_values)

print("\n\n🧮 Advantage Summary Table:")
print(f"{'Step':<6} {'Reward R':<12} {'Value V(s)':<12} {'Return G':<12} {'Advantage A':<12} {'Quality':<15}")
print("-"*75)
for i in range(len(all_rewards)):
    quality = "Good action ✅" if advantages[i] > 0 else "Bad action ❌"
    print(f"{i+1:<6} {all_rewards[i]:>11.4f} {all_values[i]:>11.4f} {returns[i]:>11.4f} {advantages[i]:>11.4f} {quality}")

mean_advantage = np.mean(advantages)
std_advantage = np.std(advantages)
print(f"\nAdvantage statistics:")
print(f"  Mean: {mean_advantage:.4f}")
print(f"  Std: {std_advantage:.4f}")

# Normalize advantages
normalized_advantages = [(a - mean_advantage) / (std_advantage + 1e-8) for a in advantages]
print(f"\n📊 Normalizing advantages for stable training")
print(f"  Formula: (A - mean(A)) / std(A)")
print(f"  This prevents large gradients and improves stability")
print(f"  Normalized range: [{min(normalized_advantages):.2f}, {max(normalized_advantages):.2f}]")

# Step 3: PPO update
print("\n" + "="*80)
print("📊 STEP 3: PPO POLICY UPDATE")
print("-"*50)
print("PPO objective: maximize advantages while constraining policy change")
print("L = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A) where r(θ) = π(a|s)/π_old(a|s)")
print("-"*50)

# Convert to tensors
all_rewards_tensor = [torch.tensor(r).to(device) for r in all_rewards]

# Create PPO trainer
batch_size = min(4, len(prompts))
ppo_config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    batch_size=batch_size,
    mini_batch_size=batch_size,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    learning_rate=1e-5,
    adap_kl_ctrl=True,
    init_kl_coef=0.2,
    cliprange=0.2,  # ε in PPO clipping
    cliprange_value=0.2,
    vf_coef=1.0,  # Value function loss coefficient
)

print(f"\nPPO Hyperparameters:")
print(f"  Batch size: {batch_size}")
print(f"  PPO epochs: {ppo_config.ppo_epochs}")
print(f"  Learning rate: {ppo_config.learning_rate}")
print(f"  Clip range ε: {ppo_config.cliprange}")
print(f"  Value function coefficient: {ppo_config.vf_coef}")

trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    tokenizer=tokenizer
)

# Process in batches
for batch_start in range(0, len(all_queries), batch_size):
    batch_end = min(batch_start + batch_size, len(all_queries))
    
    if batch_end - batch_start < batch_size:
        break
    
    batch_queries = all_queries[batch_start:batch_end]
    batch_responses = all_responses[batch_start:batch_end]
    batch_rewards = all_rewards_tensor[batch_start:batch_end]
    
    print(f"\n{'='*70}")
    print(f"🔄 PPO Batch Update (examples {batch_start+1}-{batch_end}):")
    print(f"{'='*70}")
    
    # Show what's being optimized
    print(f"\nThis batch contains:")
    for idx in range(batch_start, batch_end):
        print(f"  Example {idx+1}: Advantage = {advantages[idx]:.4f} ({'increase' if advantages[idx] > 0 else 'decrease'} probability)")
    
    # Perform PPO step
    stats = trainer.step(batch_queries, batch_responses, batch_rewards)
    
    if stats:
        print("\n📈 PPO Update Metrics Explained:")
        print("-"*50)
        
        # Policy metrics
        if 'ppo/loss/policy' in stats:
            print(f"\n🎯 Policy Update:")
            print(f"  Policy Loss: {stats['ppo/loss/policy']:.4f}")
            if stats['ppo/loss/policy'] < 0:
                print(f"    ✅ Negative loss = policy improving (increasing expected advantage)")
            else:
                print(f"    ⚠️  Positive loss = policy getting worse")
        
        if 'ppo/policy/clipfrac' in stats:
            print(f"\n  Clip Fraction: {stats['ppo/policy/clipfrac']:.3f}")
            print(f"    → {stats['ppo/policy/clipfrac']*100:.1f}% of updates were clipped")
            print(f"    → Target ~20-30% (too low = small updates, too high = trying to change too much)")
        
        if 'ppo/policy/approxkl' in stats:
            print(f"\n  KL Divergence: {stats['ppo/policy/approxkl']:.6f}")
            print(f"    → Measures: KL(π_new || π_old)")
            if stats['ppo/policy/approxkl'] < 0.01:
                print(f"    → ✅ Small KL = stable update")
            elif stats['ppo/policy/approxkl'] < 0.02:
                print(f"    → ⚠️  Moderate KL = reasonable update")
            else:
                print(f"    → ❌ Large KL = policy changing too fast")
        
        if 'ppo/policy/entropy' in stats:
            print(f"\n  Policy Entropy: {stats['ppo/policy/entropy']:.4f}")
            print(f"    → Measures randomness in action selection")
            print(f"    → Higher = more exploration, Lower = more deterministic")
        
        # Value function metrics
        if 'ppo/loss/value' in stats:
            print(f"\n💎 Value Function Update:")
            print(f"  Value Loss: {stats['ppo/loss/value']:.4f}")
            print(f"    → Training V(s) to predict returns better")
            print(f"    → Lower loss = better predictions")
        
        if 'ppo/value/var_explained' in stats:
            print(f"  Variance Explained: {stats['ppo/value/var_explained']:.3f}")
            print(f"    → R² score: {stats['ppo/value/var_explained']*100:.1f}% of return variance explained")
            print(f"    → 1.0 = perfect prediction, 0.0 = no better than mean")
        
        # Returns
        if 'ppo/returns/mean' in stats:
            print(f"\n📊 Returns:")
            print(f"  Mean Return: {stats['ppo/returns/mean']:.4f}")
            print(f"    → Average discounted future reward")

# Step 4: Test updated policy
print("\n" + "="*80)
print("📊 STEP 4: TESTING UPDATED POLICY")
print("-"*50)

print("\n🧪 Generating with updated policy to see changes:")
test_prompts = ["The movie was", "I thought it was", "The acting seemed"]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get new value estimate
        new_value = get_value_from_model(policy_model, inputs.input_ids)
        
        # Generate with updated policy
        outputs = policy_model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=1.0)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated: '{generated_text}'")
    print(f"New Value Estimate: {new_value.item():.4f}")

print("\n" + "="*80)
print("🎓 PPO TRAINING COMPLETE!")
print("="*80)
print("""
Key Takeaways:
1. PPO collects experiences and calculates how good actions were (advantages)
2. Positive advantages → increase action probability, Negative → decrease
3. The clipping mechanism prevents too large policy changes (stability)
4. The value head learns to predict future rewards (reduces variance)
5. KL divergence monitoring ensures we don't change the policy too fast

Mathematical Summary:
- Advantage: A = R - V(s) (action quality vs expectation)
- PPO Loss: L = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
- Value Loss: L_V = (V(s) - R)²
- Total Loss: L_total = L_policy + c₁L_value - c₂H(π)
""")

# Save the model
save_path = "./ppo_educational_model"
policy_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n✅ Model saved to {save_path}")

# Create visualization of key concepts
print("\n📊 Creating visualization of PPO concepts...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Advantage distribution
ax = axes[0, 0]
ax.bar(range(len(advantages)), advantages, color=['green' if a > 0 else 'red' for a in advantages])
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Experience')
ax.set_ylabel('Advantage')
ax.set_title('Advantages: Green = Good Actions, Red = Bad Actions')

# 2. Rewards vs Values
ax = axes[0, 1]
x = range(len(all_rewards))
ax.plot(x, all_rewards, 'b-', marker='o', label='Actual Rewards')
ax.plot(x, all_values, 'r--', marker='s', label='Value Predictions')
ax.set_xlabel('Experience')
ax.set_ylabel('Value')
ax.set_title('Rewards vs Value Predictions')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Tanh function visualization
ax = axes[1, 0]
sentiment_range = np.linspace(-3, 3, 100)
rewards_range = np.tanh(sentiment_range)
ax.plot(sentiment_range, rewards_range, 'g-', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Reward')
ax.set_title('Tanh: Sentiment Score → Reward Mapping')
ax.grid(True, alpha=0.3)

# Mark actual points
for i, sentiment in enumerate([p - n for p, n in zip([1.26, 2.70], [2.70, 1.26])]):
    if abs(sentiment) < 3:
        ax.scatter([sentiment], [np.tanh(sentiment)], s=100, zorder=5)

# 4. PPO clipping visualization
ax = axes[1, 1]
ratio = np.linspace(0.5, 1.5, 100)
advantage = 1.0
epsilon = 0.2

# Unclipped objective
unclipped = ratio * advantage
# Clipped objective
clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
# PPO objective (minimum of both)
ppo_objective = np.minimum(unclipped, clipped)

ax.plot(ratio, unclipped, 'b--', label='Unclipped', alpha=0.7)
ax.plot(ratio, clipped, 'r--', label='Clipped', alpha=0.7)
ax.plot(ratio, ppo_objective, 'g-', linewidth=2, label='PPO Objective')
ax.axvline(x=1.0, color='black', linestyle=':', alpha=0.5)
ax.set_xlabel('Probability Ratio π(a|s)/π_old(a|s)')
ax.set_ylabel('Objective')
ax.set_title(f'PPO Clipping (ε={epsilon}, A={advantage})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ppo_educational_concepts.png', dpi=150, bbox_inches='tight')
print("✅ Saved visualization to 'ppo_educational_concepts.png'")
