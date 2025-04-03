

::: {.cell .markdown}

## Launch containers

Inside the SSH session, bring up the Flask, FastAPI, Prometheus, and Grafana services:


```bash
# runs on node-eval-online
docker compose -f eval-online-chi/docker/docker-compose-prometheus.yaml up -d
```

Run

```bash
# run on node-eval-online
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `4-eval_online.ipynb` notebook to continue.

:::

