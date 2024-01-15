(ns clj-ppl.core)

(defn foo
  "I don't do a whole lot."
  [x]
  (println x "Hello, World!"))



(comment
  ;; probabilistic inference describes rational 
  ;; reasoning under uncertainty. 

  ;; ppls enable expressive and complex probabilistic models
  ;; to be built using abstraction and composition. 
  ;; just like prog langs really benefit from these things, 
  ;; statistical modeling does as well! 

  ;; a ppl does two things: 
  ;; 1. provides primitives for building complex prob distributions
  ;; 2. provides inference algorithms for probabilistic reasoning over 
  ;; an arbitrary program. 

  ;; if we view the semantics of the underlying deterministic language we'll 
  ;; be writing our ppl in as a map from Programs --> Executions of Programs, then 
  ;; the semantics of the ppl will be a map from Programs --> Distributions over Programs.


  ;; the main engineering challenge is an efficient impl of inference. 
  ;; could brute force by exploring all ex paths. We'll explore this first, but it will
  ;; quickly become clear that this is not practical in more complex cases. We'll then switch
  ;; over to approximate inference techniques like particle filtering. 
  )

(comment
  ;; a distribution type (constructor) takes params and returns a distribution. 
  ;; each dist d has:
  ;; 1. a method d.sample that returns a sample from the distribution, 
  ;; 2. a method d.score that returns the log probability of a possible sampled value, 
  ;; 3. optionally, a method d.support that returns the support of the distribution. 
  ;; d.sample should never be called directly. Instead, use sample(...)


  ;; we also want to be able to perform marginalization; i.e. weight return values in some way. 
  ;; see http://dippl.org/chapters/02-webppl.html for example. 
  ;; could be used to condition on data by saying weight this with 1 if it agrees w observation, else -1 
  ;; or something.  (factor(...))
  )


(comment
  ;; all inference techniques explore the space of executions of a random computation in one way or another. 
  ;; let's enumerate ALL ex paths as a first pass at this. 

  ;; sample + factor are side-processes. 
  ;; let's implement a fake version of sample that just chooses the first element from a distribution's supports. 

  (defmulti support
    (fn [dist]
      (:name dist)))
  (defmethod support :bernoulli
    [_]
    (take-while #(< %  2) (range)))
  ;; example using multimethods for defining support 
  (support {:name :bernoulli})

  (defn _sample
    [dist]
    (first (support dist)))

  (_sample {:name :bernoulli})

  (defn binomial [] (let [flip {:name :bernoulli}
                          a (_sample flip)
                          b (_sample flip)
                          c (_sample flip)]
                      (+ a b c)))
  (defn explore-first [computation] (computation))
  (explore-first binomial)


  ;; this goes back and forth between the binomial computation AND the randomness handling (_sample)
  ;; fns, but it only allows us to look at a single ex path. WE want to be able to return from _sampel
  ;; multiple times with different values. We can't do this with an ordinary function return; we need an 
  ;; explicit handle to the return context. We need to be able to specify the future of the computation 
  ;; from the point that sample is called. This is known as a *continutation*. 

  ;; Let's practice w continuations. 

  (defn square [x] (* x x))
  (print (square 3))
  ;; what happens AFTer we return 3 * 3? we print. we can express this as a continuation of square: 

  (defn cps-square [cont x] (cont (* x x)))

  (cps-square print 3)

  ;; cps is a way of writing programs so that the "next" action that will be taken with the current 
  ;; expression is ALWAYS available on the stack. 
  ;; functions never really "return" in cps. they only call other continutations with values that 
  ;; they WOULD have returned.

  (defn factorial [x]
    (if (= 1 x) 1 (* x (factorial (- x 1)))))

  (defn cps-factorial [cont x]
    (if (= x 0)
      (cont 1)
      (let
       ;; note that we construct a new continuation here 
       ;; for the computation * x rest bc it comes AFTER the recursive call. 
       [new-cont (fn
                   [rest-of-factorial]
                   (cont (* x rest-of-factorial)))]
        (cps-factorial new-cont (- x 1)))))

  (cps-factorial print 5)

  ;; the continuation turns application inside out. normally, the * op would be on 
  ;; the OUTSIDE of the computation. Instead, it gets pushed INSIDE the function in 
  ;; the form of new-cont! 

  ;; let's rewrite binomial in cps: 

  (defn cps-_sample
    [cont dist]
    (cont (_sample dist)))



  (defn cps-binomial [cont]
    (cps-_sample
     (fn [a]
       (cps-_sample
        (fn [b] (cont (+ a b)))
        {:name :bernoulli}))
     {:name :bernoulli}))

  ;; I removed c here for clarity. here's the narrative. 

  ;; 1. We sample a, then pass a to a continuation. 
  ;; 2. that continuation takes a and samples b, then passes that to a continuation. 
  ;; 3. that continuation adds a and b together, then passes the result to the original continuation
  ;; provided to cps-binomial. 

  ;; EVERY STEP MUST HAVE A CONTINUATION. this will help guide you when writing cps. 

  ;; Now let's write the exploration. 
  ;; we want to provide sample with the continuation of WHERE it is called, and then call this continuation 
  ;; several times with different values. This pattern of a function that receives a continuation from the main 
  ;; computation + returns only by calling that continuation is called a *coroutine*. 

  (def unexplored-futures (atom []))
  (def return-vals (atom []))


  (defn _sample [cont dist]
    (let [supp (support dist)]
      ;; push each continued support value onto the stack of 
      ;; unexplored possibilities
      (doseq [x supp] (swap! unexplored-futures conj (fn [] (cont x))))
      (let [next (last @unexplored-futures)]
        (swap! unexplored-futures drop-last)
        (next) ;; try the the next val
        )))

  (defn exit [return-val]
    (swap! return-vals conj return-val)
    (when (not-empty @unexplored-futures)
      (let [next (last @unexplored-futures)]
        (swap! unexplored-futures drop-last) ;; pop
        (next) ;; do next thing 
        )))

  (defn explore [cps-computation]
    (cps-computation exit) ;; accumulate return vals over ex paths
    @return-vals)

  (defn cps-binomial [cont]
    (_sample
     (fn [a]
       (_sample
        (fn [b]
          (_sample
           (fn [c]
             (cont (+ a b c)))
           {:name :bernoulli}))
        {:name :bernoulli}))
     {:name :bernoulli}))

  (explore cps-binomial)

  
  ;; This keeps track of all executions of the computation,
  ;; but it doesn't know anything about probabilities. In order to do this, 
  ;; we need to compute scores... 

  ;; TODO: step through above code in debugger to make sure you really understand 
  ;; what is happening there. 










  )



