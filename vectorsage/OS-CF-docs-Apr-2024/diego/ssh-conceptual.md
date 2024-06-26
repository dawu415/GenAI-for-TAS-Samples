# Cloud Foundry app SSH components and processes
This topic tells you about the Cloud Foundry SSH components that are used for access
to deployed app instances. Cloud Foundry supports native SSH access to apps and load balancing of SSH sessions with the load balancer for your Cloud Foundry deployment.
For procedural and configuration information about app SSH access, see [SSH Overview](https://docs.cloudfoundry.org/devguide/deploy-apps/app-ssh-overview.html).

## SSH components
Cloud Foundry SSH includes two central components: an implementation of an SSH proxy server and a lightweight SSH daemon. If these components are deployed and configured correctly, they provide a
simple and scalable way to access containers apps and other long-running processes (LRPs).

### SSH daemon
SSH daemon is a lightweight implementation that is built around the Go SSH library. It supports command execution, interactive shells, local port forwarding, and secure copy. The daemon is self-contained and has no dependencies on the container root file system.
The daemon is focused on delivering basic access to app instances in Cloud Foundry. It is intended to run as an unprivileged process, and interactive shells and commands run as the daemon user. The daemon
only supports one authorized key, and it is not intended to support multiple users.
The daemon is available on a file server and Diego LRPs that want to use it can include a
download action to acquire the binary and a run action to start it.
Cloud Foundry apps download the daemon as part of the lifecycle bundle.

### SSH proxy authentication
The SSH proxy hosts the user accessible SSH endpoint and is responsible for authentication,
policy enforcement, and access controls in the context of Cloud Foundry.
After you successfully authenticate with the proxy, the proxy attempts to locate
the target container and create an SSH session to a daemon running inside the container.
After both sessions have been established, the proxy manages the communication between your SSH client
and the container SSH Daemon.