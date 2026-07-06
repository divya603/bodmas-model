# 👩‍💻 Setup notes

**You never have to do these steps they are just documented so you know how we
got to the initial project structure. Also include some useful notes about
things.**

## How the project was initialized

The project starts off with a bunch of files which you might find unfamiliar.
There is no magic there though. The basic directory structure was automatically
created by [Vite](https://vitejs.dev).

First, you need to install Node.js on your computer if you haven't already. You
can download the latest version [here](https://nodejs.org/en/download/). Verify
that you have the `npm` command in your terminal program of choice.

Next, I initialized a default Vite project:

```sh
mkdir smile
cd smile
npm create vite@latest .
```

It asked me for a project name ("smile"), a framework ("vue"), and a variant
("vue" meaning vanilla javascript, not typescript). Then you just run `npm i` to
install all the dependencies.

Next, I installed the documentation system,
[Vitepress](https://vitepress.vuejs.org). From inside the smile folder:

```sh
npm add vitepress --save-dev
mkdir docs && echo `# Smile` > docs/index.md
```

Then I added these commands to the `package.json` file:

```json
{
  ...
  "scripts": {
    "docs:dev": "vitepress dev docs",
    "docs:build": "vitepress build docs",
    "docs:serve": "vitepress serve docs"
  },
  ...
}
```

To run the documentation locally then I just type `npm run doc:dev` and open the
provided localhost link. The documentation on how to configure Vitepress is
[here](https://vitepress.vuejs.org).

## Node package manager (npm)

The project assumes that you are using node package manager (npm). Generally all
commands involve `npm <something>`.

An important file here is `package.json` which describes the javascript
dependencies of the current project but also provides a set of commands
available for managing the project. The content contains something like this:

```json
{
  ...
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "docs:dev": "vitepress dev docs",  // this is a docs command!
    "docs:build": "vitepress build docs", // this too
    "docs:serve": "vitepress serve docs" // this too
  },
  ...
}
```

In the `scripts` section, you see several commands. To run them you just type
`npm run <cmd>`. For instance typing `npm run docs:dev` will effectively run the
command `vitepress dev docs`.  
The `npm run <cmd>` commands are just aliases/shorthands for running build steps
of the vitepress documentation system, and later for the Smile project itself.
It is similar to a Makefile. Anytime you forget the possible commands just type
`npm run` on a line by itself and you'll get a listing like this:

```
➜ npm run
Scripts available in smile@0.0.0 via `npm run-script`:
  dev
    vite
  build
    vite build
  preview
    vite preview
  docs:dev
    vitepress dev docs
  docs:build
    vitepress build docs
  docs:serve
    vitepress serve docs
```

## GitHub Actions - Vitepress Docs

The script which runs the deploy action is located at
`.github/workflows/docs-deploy.yml`. This relies on several "secrets" which are
configured in the GitHub website's repository settings (these are environment
variables hidden from the public GitHub repo since they contain sensitive
information). A few other notes:

- This is the [rsync package](https://github.com/Burnett01/rsync-deployments)
  used.
- I also found [this guide](https://zellwk.com/blog/github-actions-deploy/)
  helpful.
