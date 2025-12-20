---
title: "Learning Machine Learning by Building It From Scratch in Rust"
description: "How I went from using ML libraries to actually understanding gradient descent and simple language models by implementing everything myself in Rust."
pubDate: 2025-12-15
tags: ["Rust", "Machine Learning", "AI", "From Scratch", "Statistics"]
draft: false
---

For a while, I had been using machine learning without really understanding it.
Libraries made it easy to train models and get results, but the mechanics were hidden behind layers of abstraction.

So instead of starting with a framework, I went all the way back to basics — **paper and math** — and rebuilt the foundations myself.

---

## Stripping ML Down to Its Core

You define:

- a model (some function with parameters),
- a loss function (how wrong the model is),
- and a rule for updating parameters to reduce that loss.

Everything else builds on top of this.

To make that concrete, I started with the simplest possible model.

---

## Linear Regression, Actually Understood

Imagine you have some data points on a graph — maybe house sizes and their prices. You want to find a straight line that best predicts the price given the size.

### The Model: A Straight Line

Any straight line can be written as:

$$
\hat y = wx + b
$$

In plain English:

- $x$ is your **input** (e.g., house size)
- $\hat y$ (read "y-hat") is your **prediction** (e.g., predicted price)
- $w$ is the **weight** — it controls how steep the line is (how much $y$ changes when $x$ changes)
- $b$ is the **bias** — it shifts the line up or down (the value of $y$ when $x$ is zero)

The whole game is figuring out what values of $w$ and $b$ make the line fit the data best.

### Measuring How Wrong We Are

Given a real data point $(x, y)$ where $y$ is the actual correct answer, the **error** is just:

$$
y - \hat y
$$

This tells us how far off our prediction was. Positive means we guessed too low, negative means too high.

But we don't just want to know the error — we want a single number that tells us how bad the model is overall. That's the **loss function**. We use the squared error:

$$
L = (y - \hat y)^2
$$

Why squared? Two reasons:

1. It makes all errors positive (a prediction that's 5 too high is just as bad as 5 too low)
2. It punishes big errors more than small ones (being off by 10 is worse than being off by 2, and squaring makes that difference dramatic)

### Finding the Right Direction to Improve

Here's the key insight: if we could figure out **which direction to nudge $w$ and $b$** to make the loss smaller, we could just keep nudging until the line fits perfectly.

That's what **gradients** tell us. A gradient is just the answer to: "If I increase this parameter a tiny bit, does the loss go up or down, and by how much?"

The math gives us:

$$
\frac{\partial L}{\partial w} = -2x(y - \hat y)
$$

$$
\frac{\partial L}{\partial b} = -2(y - \hat y)
$$

Here's what the **symbols** mean:

- $\frac{\partial L}{\partial w}$ is "how much does the loss change when we change $w$"
- If this number is **positive**, increasing $w$ makes the loss worse → we should decrease $w$
- If this number is **negative**, increasing $w$ makes the loss better → we should increase $w$

Same logic applies to $b$.

### The Update Rule: Learning Step by Step

Now we can actually improve the model. Each step, we nudge the parameters in the direction that reduces the loss:

$$
w \leftarrow w - \alpha \frac{\partial L}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial L}{\partial b}
$$

The $\leftarrow$ just means "update the value." And $\alpha$ (alpha) is the **learning rate** — a small number (like 0.01) that controls how big each step is.

Why subtract? Because if the gradient is positive (loss goes up when $w$ increases), we want to go the _opposite_ direction and decrease $w$. The subtraction handles that automatically.

**That's it.** Repeat this process thousands of times, and the line gradually converges to the best fit.

I worked through this by hand for a single data point.
After one update step, the model landed exactly on the target.

---

## Turning the Math Into Rust

Once the math was clear, writing the code was almost boring — in a good way.

The Rust version was just:

- compute the prediction,
- compute the gradients,
- update `w` and `b`,
- repeat.

No libraries. No autodiff. No magic.

What surprised me was how **fragile** learning can be.
A learning rate that worked fine for small values caused the model to completely explode when inputs were larger. Seeing numbers shoot to infinity made it obvious why normalization and careful step sizes matter.

---

## Simple Language Models (Without Neural Networks)

With gradient descent under control, I wanted to try something different: **language**.

Instead of jumping to neural networks, I built the simplest possible language models based purely on counting.

### Character-Level Bigrams

The first version learned:

$$
P(\text{next character} \mid \text{previous character})
$$

Training was just counting transitions between characters.

The output looked like word-shaped noise — not broken, just extremely limited.

### Word-Level Bigrams

Switching from characters to **words** made a huge difference:

$$
P(\text{next word} \mid \text{previous word})
$$

With enough training text, the model started producing sentences that felt oddly familiar — grammatically plausible, topical, and clearly inspired by the source material.

Still wrong. Still shallow.
But undeniably _language-like_.

That made one thing very clear: model quality depends heavily on **what information the model is allowed to remember**.

---

## Source Code

All of this lives here:

👉 **neural-nets**
[https://github.com/Lucas8448/neural-nets](https://github.com/Lucas8448/neural-nets)

The repository contains:

- linear regression implemented from scratch,
- gradient descent experiments,
- character- and word-level language models
