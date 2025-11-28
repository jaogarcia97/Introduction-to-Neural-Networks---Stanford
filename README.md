# Introduction-to-Neural-Networks---Stanford
These are the notes for Serena Young's Lecture : https://www.youtube.com/watch?v=d14TUNcbn1k

9:31 
Question: What does find the effect of y on f mean?
Answer: That is just translated to df/dy
Additional notes: What she said on 9:57, if you change y a little bit, q will change a little bit just means that if you change dy, the overall df/dy will also change since you changed its denominator dy.
This also appeared in 10:15


19:41
Question: Why is the value of the gradient 1? (Referring to first red number from the right, in back propagation)
Answer: df/df = 1


21:46 
Question: Why -0.53 and not 1? : Referring to the 2nd -0.53 from the right
![alt text](image-1.png)
Answer:
The local gradient is indeed 1, however, the red number represesnts the global gradient (hoe much the final output changes)

To find this, you must multiple the upstream gradient (coming from the right which is the 1st -0.53) by the local gradient (1). You multiply based on the logic of "compounding" the change from the first onto the next.

23:18:
Question: How did she get f_a((x) = ax?
Answer: She mentioned that "a" is just a constant she used that would represent -1,
ax is basically just:

-1 * what ever you answer you came up with the previous nodes;

wherein the "what ever you came up with the previous nodes" is -w0x0+w1x1+w2. 


21:48
Question: Why is the local gradient 1? in [1]x[0.2]=0.2
what did she mean in 23:53 that if we have an addition node the gradient with respect to each of the inputs to the addition is just going to be "one"

Answer: Refer back to q = x + y in 10:18. you want to know the effect of x on q, you need to solve for dq/dx, and so on. 

24:49
Question: How did they get 2 and -1 as local gradients?
Answer: Refer again to 10:18 and look at the multiplication part f = qz

31:06
Such a beauty.


33:16
Question: How does the max node work?
Answer: Only the value that has affected "or won" ove got passed down the rest of the computational graph, since it is the only one that affected change, it only makes sense when we're passing our gradients back we just want to adjust who "won" (hence the max)




### Introduction

Every node contains a single "formula" that requires an input. This formula is what you call a **Node Operation**. You get this "formula" for each node by breaking down the formula of choice into smaller pieces

For example, if your formula of choice is (x+y)z. This means that you are creating a neural network under this (x+y)z architecture. 

This would then be broken down into nodes that would compose of the basic operations as if there are steps

Node Operation for Node 1: $x+y$

Node Operation for Node 2 : (whatever the output of Node 1 is) $* z$

The result from the **Node Operation** is what you call a **Forward Pass Value**. In the video, these are the green numbers above the links from one node to another. These neural links are what you call **Edges**. 

The whole neural network is just a *computational graph* of your formula of choice. 

We use the formulas for these node operations in order to determine how much change we must adjust our initial values.

### Gates

#### 1. The Addition Gate (Addition Node)
* **Node Operation:** $f(x, y) = x + y$
    - Add x and add y together, you get the value for for the forward pass
* **Local Gradient Computation**: 
    - For the Edge of x
        - $df/dx = 1 + 0$
    - For the Edge of y
        - $df/dy = 0 + 1$
* **Global Gradient Computation**: 
    - Global Gradient = Local Gradient * Upstream Gradient 
    - Resultant Grobal Gradient For the Edge of x
        - $ [df/dx] × UpstreamGradient$
    - Resultant Global Gradient For the Edge of y
        - $ [df/dy] × UpstreamGradient$

* Reflection: This is why the Add Gate is called a Gradient Distributor since it just copies whatever the Upstream Gradient is and passes it (or "distributes it") back to the nodes it came from


#### 2. The Multiplication Node ($*$)
* **Function:** $f(x, y) = x \cdot y$
* **How to read it:** "Multiply the top wire ($x$) and the bottom wire ($y$)."
* **Derivative Rule (Gradient Switcher):**
    * The derivative of $x$ becomes $y$.
    * The derivative of $y$ becomes $x$.
    * *Intuition:* You multiply the upstream gradient by the value of the *other* input wire.

#### 3. The Max Node (max)
* **Function:** $f(x, y) = \max(x, y)$
* **How to read it:** "Output whichever number is bigger."
* **Derivative Rule (Gradient Router):**
    * **For the Winner:** Gradient is **1** (it passes the upstream gradient through).
    * **For the Loser:** Gradient is **0** (it blocks the gradient).