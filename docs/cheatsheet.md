# Cheat sheet

## Interacting with project

### Creating a new project

Called `my_cool_project` for GitHub user `ghuser` (assumes you already have the
.env files for your lab or configuration):

```
gh repo create my_cool_project --private --template nyuccl/smile
gh repo clone ghuser/my_cool_project
gh repo edit ghuser/my_cool_project --description "my new project description"
cd my_cool_project
npm run config:upload
npm run setup_project
```

### Delete a repo from the command line

You may have to follow instructions to enable permissions to delete the repo)
and the local files:

```
gh repo delete ghuser/my_cool_project
rm -rf my_cool_project
```

### Run development server

```
npm run dev
```

### Run development server for documentation

Only when working on the `nyuccl/smile` repo

```
npm run docs:dev
```

### Build the local files and serve them

This mimics the final deployment state of the files

```
npm run build
npm run serve
```

### Deploy a specific version of your code to the server

Replace `XXXXX` with the git hash for the version you want to deploy

```
gh workflow run deploy.yml -f github_sha=XXXXX
```
