# Penn's talks

###### 2025

- `zh+en` [MCTS Variant Performance Prediction](./2025-12-14) - Class Presentation Songshan Lake 2025

# Template

<div align="center"><a href="https://github.com/LittleSound/talks-template?tab=readme-ov-file">Origin repo</a></div>

## Features

- ⚡️ [Slidev](https://github.com/slidevjs/slidev) - write slides in markdown
- 🍱 Manage all your slides in a repo
- 🎯 [picker](https://github.com/littlesound/picker) - Choose a repository in the terminal to execute the script
- 📱 Write interactive slides with [Vue](https://vuejs.org/)
- 📦 [pnpm](https://pnpm.io/) - package manager
- 😃 [Use icons from any icon sets with classes](https://github.com/antfu/iconify-json)

## Try it now!

### GitHub Template

[Create a repo from this template on GitHub](https://github.com/LittleSound/talks-template/generate).

### Clone to local

If you prefer to do it manually with the cleaner git history

```bash
npx degit LittleSound/talks-template my-talks
cd my-talks
pnpm i # If you don't have pnpm installed, run: npm install -g pnpm
```

## Checklist

When you use this template, try follow the checklist to update your info properly

- [ ] Change the author name in `LICENSE`
- [ ] Remove the `.github` folder which contains the funding info
- [ ] Use `README-template.md` to replace `README.md`
- [ ] Copy the `0000-00-00` folder and start creating your actual talk
- [ ] Find the TODO tags in the file to learn more

And, enjoy :)

## Usage:

### Development

```bash
pnpm dev
```

visit <http://localhost:3030>

Edit the `<your talk folder>/src/slides.md` to see the changes.

Learn more about Slidev at the [documentation](https://sli.dev/).

### Build

To build the Slides Website, run

```bash
pnpm build
```

### Export to PDF

To export the Slides to PDF, run

```bash
pnpm export
```

the PDF will be generated in the `<your talk folder>/slides.pdf` folder.

### Development with Remote Host

To develop the Slides Website with a remote host, run

```bash
pnpm dev:host
```
