# IMGGEN

> This project is a fastapi wrapper around a pretrained stable diffusion model.

## Setup

### Requirements

- [docker](https://www.docker.com/products/docker-desktop/)
- [docker-compose](https://docs.docker.com/compose/install/)

### Running Locally

- clone git repo
- enter project directory
- run `docker compose up`

## Usage

- post to `http://localhost:8069/prompt`
  - with query string `?prompt={prompt}`
  - or with body:
  ```json
  {
    "prompt": "",
  }
  ```
