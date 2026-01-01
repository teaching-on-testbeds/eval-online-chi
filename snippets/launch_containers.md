

::: {.cell .markdown}

## Launch containers

Inside the SSH session, bring up the Flask, FastAPI, Prometheus, and Grafana services:


```bash
# runs on node-eval-online
docker compose -f eval-online-chi/docker/docker-compose-prometheus.yaml up -d
```

To access the Jupyter service, we will need its randomly generated secret token (which secures it from unauthorized access). We'll get this token by running `jupyter server list` inside the `jupyter` container:

```bash
# run on node-eval-online
docker exec jupyter jupyter server list
```

Look for a line like

```
http://localhost:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of `localhost`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `4-eval_online.ipynb` notebook to continue.

:::
