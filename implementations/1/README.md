# Implementation #1: SDD + Antigravity

## Introduction

In order to push the project's AI-assisted philosophy as much as possible, the specifics were not written by hand. Rather, I sent the [design document](./Retriva-SDD_implementation_1.pdf) to MS Copilot so that it could write the spec files in accordance with Antigravity's conventions (interestingly, I had created the design document by brainstorming with Copilot itself!).

For security reasons, I installed a VirtualBox virtual machine running Ubuntu 24.04 on my development host — which runs Ubuntu 24.04 as well — and used Antigravity inside it. In this way, I sandboxed the agentic tool to prevent it from unintentionally compromising the host operating system. I used a condivisa directory to exchange files between the host and the guest:

```
# Shared folder seen from the guest machine:
llandre@vm-ubnt-24-04-4:~$ mount | grep vboxsf
implementation on /mnt/shared/implementation type vboxsf (rw,nodev,relatime,iocharset=utf8,uid=0,gid=984,dmode=0770,fmode=0770,tag=VBoxAutomounter)
```

## Agents at work

As usual, Copilot was zelous, therefore it has also produced instructions on how to use the ready-to-drop SDD pack it generated:

```
How to use it in Antigravity

Unzip the package into your project root so the .agent/ folder sits at workspace level. Community Antigravity workflow packs rely on .agent/workflows, and the slash commands are detected from those Markdown workflow files.

Open the workspace in Antigravity and start a Planning conversation; Antigravity recommends Planning mode for deeper, more complex work that should produce reviewable artifacts.
Run the workflows in this order:
* /define Finalize specs/001-qa-only-poc using the existing design package.
* /architect Prepare the approved implementation plan for specs/001-qa-only-poc.
* /execute Implement only the approved tasks in specs/001-qa-only-poc/tasks.md.
* /verify Validate specs/001-qa-only-poc against its acceptance criteria.
```

I followed the suggested flow without any particular issues.

This is how I configured the agent:

![](assets/screenshot_01.png)

Occasionally, I experienced this error, but I always solved it by triggering the operation again:

![](assets/screenshot_02.png)

Sometimes, I interacted with the agents during the execution of the workflow:

![](assets/screenshot_03.png)

This is the last step of the workflow:

![](assets/screenshot_04.png)

At some point in the workflow, I also executed some tests prepared by Antigravity to verify that what was coded until then was OK. I had to fix a couple of things to make test work, of course with the help of the conversational agent:

![](assets/screenshot_05.png)
