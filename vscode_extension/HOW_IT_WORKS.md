# How the VSCode Extension Works

A simple explanation of what this extension does and how it's built.

## What it does

While you're typing Python code, the extension quietly asks a locally running AI model
"what do you think comes next?" and shows the answer as a normal autocomplete suggestion,
right alongside VSCode's built-in ones.

## Tech stack

| Part | Technology |
|---|---|
| Extension code | TypeScript, compiled to JavaScript with `tsc` |
| Extension platform | VSCode Extension API (`vscode` package) |
| Talking to the model | Node.js built-in `http` module |
| Model server | Python + Flask |
| The model itself | A Transformer, trained with PyTorch |
| Package management | npm |

## How a suggestion actually happens

1. **You type a character** in a `.py` file.
2. VSCode asks every registered completion provider "got anything?" — including this
   extension's provider ([`extension.ts`](src/extension.ts), registered for the `python`
   language).
3. The extension grabs the **last ~20 lines of code** up to your cursor as context.
4. It sends that text as JSON to a Python server running on your machine
   (`POST http://localhost:5000/autocomplete`).
5. The Python server (`serve.py`, in the main project) feeds that text into the trained
   model, which predicts a few likely continuations along with how confident it is in each.
6. The server sends back up to 5 candidate completions.
7. The extension does some cleanup on them — removing duplicates, cutting text off at the
   next newline, trimming anything that would clash with text already after your cursor —
   and hands the results to VSCode as completion items.
8. VSCode shows them in the normal suggestion dropdown, mixed in with everything else
   (like Pylance's suggestions).

## Why it's split into two parts

The actual AI model runs in **Python** (that's what PyTorch models are built with), but
VSCode extensions can only be written in **JavaScript/TypeScript**. So the project runs the
model as a small local web server, and the extension is just a thin client that asks that
server for predictions over HTTP — nothing AI-related happens inside the extension itself.
