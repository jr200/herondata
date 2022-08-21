Thank you for agreeing to do our take-home task! This will give us the opportunity to assess your problem solving and will give you an example problem that you may be asked to solve on the job. 

**Please have a Github repository shared with @[dominickwok](https://github.com/dominickwok) and @[cesarferradas](https://github.com/cesarferradas) with any thinking / diagrams / code**. We will only evaluate what we see committed to the main / master branch in the repo at or before the 1.5 hour mark

**Context**

At Heron Data we work with a lot of bank transaction data, and below is an example list of bank transactions in json format, with the schema of the dataset

**Example Data**

[https://gist.github.com/dominickwok/3ba15527d50d6fa582895e55e21700ce](https://gist.github.com/dominickwok/3ba15527d50d6fa582895e55e21700ce)

**Schema**

- **timestamp** - `datetime` the time at which the transaction occurred
- **description** - `string` a variable length string that explains what the transaction represents
- **amount** - `float` the value of transaction

One of the ways that we add value for customers is by identifying **recurring transactions** — transactions that occur at a regular cadence. You can think of this as transactions like subscriptions (internet, phone, Netflix, Spotify), rent, or even something like a weekly company lunch.

**Problem**

In a repository that you share with @[cesarferradas](https://github.com/cesarferradas) and @[dominickwok](https://github.com/dominickwok), please do the following: 

1. In a `[README.md](http://readme.md)` file, please outline an approach to identifying recurring transactions from a set of transactions. Include any external resources used if any
2. Please implement your approach as a function
    1. i.e., `def identify_recurring_transactions(transactions: List[Transaction]) -> List[Transaction.id]`
    2. You may use any language such as, but not limited to, Python, Javascript, Golang, Rust
3. (bonus) Please include unit tests
4. (bonus) Please discuss the following:
    1. How would you measure the accuracy of your approach?
    2. How would you know whether solving this problem made a material impact on customers?
    3. How would you deploy your solution?
    4. What other approaches would you investigate if you had more time?

Please commit the code to the main / master branch of the shared Github repository for us to see within the 90 minute time limit.

Best of luck!

Team Heron

— 

Original instructions

Thank you for agreeing to do our take-home task! This will give us the opportunity to assess your problem solving and will give you an example problem that you may be asked to solve on the job. 

The 1.5hr challenge is intentionally open-ended and requires both product thinking and coding. We will be asking you to implement a function that operates on a set of [transactions](https://docs.herondata.io/api#tag/Transactions/paths/~1api~1transactions/post). We primarily use Python at Heron, but you are welcome to use any language and framework for the exercise. 

**Please have a Github repository shared with @[dominickwok](https://github.com/dominickwok) and @[cesarferradas](https://github.com/cesarferradas) with any thinking / diagrams / code**. We will only evaluate what we see committed to the main / master branch in the repo at or before the 1.5 hour mark.

**When would work for us to send you the main brief for the challenge**? Please let us know if you have questions!