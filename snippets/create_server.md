


::: {.cell .markdown}

## Launch and set up a VM instance- with python-chi

We will use the `python-chi` Python API to Chameleon to provision our VM server. 

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. 

:::

::: {.cell .code}
```python
from chi import server, context, lease, network
import chi, os, time, datetime

context.version = "1.0" 
context.choose_project()
context.choose_site(default="KVM@TACC")

username = os.getenv('USER') # all exp resources will have this prefix
```
:::

::: {.cell .markdown}

We will reserve and bring up an `m1.medium` instance with the `CC-Ubuntu24.04` disk image. 

> **Note**: the following cells bring up a server only if you don't already have one with the same name! (Regardless of its error state.) If you have a lease or a server in ERROR state already, delete it first in the Horizon GUI before you run these cells.

:::


::: {.cell .markdown}

First we will reserve the VM instance for 6 hours, starting now:

:::


::: {.cell .code}
```python
l = lease.Lease(f"lease-eval-online-{username}", duration=datetime.timedelta(hours=6))
l.add_flavor_reservation(id=chi.server.get_flavor_id("m1.medium"), amount=1)
l.submit(idempotent=True)
```
:::


::: {.cell .code}
```python
l.show()
```
:::


::: {.cell .markdown}

Now we can launch an instance using that lease:

:::

::: {.cell .code}
```python
s = server.Server(
    f"node-eval-online-{username}", 
    image_name="CC-Ubuntu24.04",
    flavor_name=l.get_reserved_flavors()[0].name
)
s.submit(idempotent=True)
```
:::

::: {.cell .markdown}

By default, all connections to VM resources are blocked, as a security measure.  We need to attach one or more "security groups" to our VM resource, to permit access over the Internet to specified ports.

The following security groups will be created (if they do not already exist in our project) and then added to our server:

:::


::: {.cell .code}
```python
security_groups = [
  {'name': "allow-ssh", 'port': 22, 'description': "Enable SSH traffic on TCP port 22"},
  {'name': "allow-5000", 'port': 5000, 'description': "Enable TCP port 5000 (used by Flask)"},
  {'name': "allow-8000", 'port': 8000, 'description': "Enable TCP port 8000 (used by FastAPI)"},
  {'name': "allow-8888", 'port': 8888, 'description': "Enable TCP port 8888 (used by Jupyter)"},
  {'name': "allow-3000", 'port': 3000, 'description': "Enable TCP port 3000 (used by Grafana)"},
  {'name': "allow-9090", 'port': 9090, 'description': "Enable TCP port 9090 (used by Prometheus)"},
  {'name': "allow-8080", 'port': 8080, 'description': "Enable TCP port 8080 (used by cAdvisor, Label Studio)"}
]
```
:::


::: {.cell .code}
```python
for sg in security_groups:
  secgroup = network.SecurityGroup({
      'name': sg['name'],
      'description': sg['description'],
  })
  secgroup.add_rule(direction='ingress', protocol='tcp', port=sg['port'])
  secgroup.submit(idempotent=True)
  s.add_security_group(sg['name'])

print(f"updated security groups: {[sg['name'] for sg in security_groups]}")
```
:::


::: {.cell .markdown}

Then, we'll associate a floating IP with the instance:

:::

::: {.cell .code}
```python
s.associate_floating_ip()
```
:::

::: {.cell .code}
```python
s.refresh()
s.check_connectivity()
```
:::

::: {.cell .markdown}

In the output below, make a note of the floating IP that has been assigned to your instance (in the "Addresses" row).

:::

::: {.cell .code}
```python
s.refresh()
s.show(type="widget")
```
:::





::: {.cell .markdown}

### Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.

:::

::: {.cell .code}
```python
s.execute("git clone https://github.com/teaching-on-testbeds/eval-online-chi")
```
:::


::: {.cell .markdown}

### Set up Docker

Here, we will set up the container framework.

:::

::: {.cell .code}
```python
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```
:::


::: {.cell .markdown}

## Open an SSH session

Finally, open an SSH sesson on your server. From your local terminal, run

```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where

* in place of `~/.ssh/id_rsa_chameleon`, substitute the path to your own key that you had uploaded to KVM@TACC
* in place of `A.B.C.D`, use the floating IP address you just associated to your instance.

:::
