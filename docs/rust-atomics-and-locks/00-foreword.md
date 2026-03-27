> **Rust Atomics and Locks** by Mara Bos (O'Reilly).
> Copyright 2023 Mara Bos. Licensed under CC BY-NC-ND 4.0.
> Source: https://mara.nl/atomics/

---


# Foreword


This book provides an excellent overview of low-level concurrency in the Rust language, including threads, locks, reference counts, atomics, mailboxes/channels, and much else besides. It digs into issues with CPUs and operating systems, the latter summarizing challenges inherent in making concurrent code work correctly on Linux, macOS, and Windows. I was particularly happy to see that Mara illustrates these topics with working Rust code. It wraps up by discussing semaphores, lock-free linked lists, queued locks, sequence locks, and even RCU.


So what does this book offer someone like myself, who has been slinging C code for almost 40 years, most recently in the nether depths of the Linux kernel?


I first learned of Rust from any number of enthusiasts and Linux-related conferences.  Nevertheless, I was happily minding my own business until I was called out by name in a Rust-related LWN article, ["Using Rust for Kernel Development"](https://lwn.net/Articles/870555/).
Thus prodded, I wrote a blog series entitled ["So You Want to Rust the Linux Kernel?"](https://paulmck.livejournal.com/62436.html) This blog series sparked a number of spirited discussions, a few of which are visible in the series' comments.


In one such discussion, a long-time Linux-kernel developer who has also written a lot of Rust code told me that when writing concurrent code in Rust, you should write it the way Rust wants you to. I have since learned that although this is great advice, it leaves open the question of exactly what Rust wants. This book gives excellent answers to this question, and is thus valuable both to Rust developers wishing to learn concurrency and to developers of concurrent code in other languages who would like to learn how best to do so in Rust.


I of course fall into this latter category. However, I must confess that many of the spirited discussions about Rust concurrency remind me of my parents' and grandparents' long-ago complaints about the inconvenient safety features that were being added to power tools such as saws and drills. Some of those safety features are now ubiquitous, but hammers, chisels, and chainsaws have not changed all that much. It was not at all easy to work out which mechanical safety features would stand the test of time, so I recommend approaching software safety features with an attitude of profound humility.  And please understand that I am addressing the proponents of such features as well as their detractors.


Which brings us to another group of potential readers, the Rust skeptics. While I do believe that most Rust skeptics are doing the community a valuable service by pointing out opportunities for improvement, all but the most Rust-savvy of skeptics would benefit from reading this book. If nothing else, doing so would enable them to provide sharper and better-targeted criticisms.


Then there are those dyed-in-the-wool non-Rust developers who would prefer to implement Rust's concurrency-related safety mechanisms in their own favorite language.  This book will give them a deeper understanding of the Rust mechanisms that they would like to replicate, or, better yet, improve upon.


Finally, any number of Linux-kernel developers are noting the progress that Rust is making toward being included in the Linux kernel; for example, see Jonathan Corbet's article, ["Next Steps for Rust in the Kernel"](https://lwn.net/Articles/908347/). As of October 2022, this is still an experiment, but one that is being taken increasingly seriously.  In fact, seriously enough that Linus Torvalds has accepted the first bits of Rust-language support into version 6.1 of the Linux kernel.


Whether you are reading this book to expand your Rust repertoire to include concurrency, to expand your concurrency repertoire to include Rust, to improve your existing non-Rust environment, or just to look at concurrency from a different viewpoint, I wish you the very best on your journey!


Paul E. McKenney


Meta Platforms Kernel Team


Meta


October 2022

