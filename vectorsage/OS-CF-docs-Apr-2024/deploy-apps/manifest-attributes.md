# App manifest attribute reference
You can manage app properties and behavior using cf CLI commands or the app manifest (a YAML properties file). This topic describes manifest formatting and provides a list of attributes available for app manifests. You can use it with [Deploying with app manifests](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest.html), which provides basic procedures and guidance for deploying apps with manifests.
For more information about V3 manifest properties, see the [Cloud Foundry API (CAPI) V3 documentation](http://v3-apidocs.cloudfoundry.org/index.html#space-manifest).

## Manifest format
Manifests are written in YAML. The following manifest illustrates some YAML conventions:

* The manifest begins with three dashes.

* The `version` property specifies a manifest schema version. This property is optional. For more information, see [Add schema version to a manifest](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest-attributes.html#manifest-schema-version).

* The `applications` block begins with a heading followed by a colon.

* The app `name` is preceded by a single dash and one space.

* Subsequent lines in the block are indented two spaces to align with `name`.
```

---
version: 1
applications:

- name: my-app
memory: 512M
instances: 2
```

**Important**
If your app name begins with the dash character (`-`), you cannot interact with the app using the cf CLI. This is because the cf CLI interprets the dash as a flag.

### Add schema version to a manifest

**Important**
This attribute is available with CAPI V3 only. To push a manifest that uses this attribute, do one of the following:

* Use cf CLI v7 or v8.
See [Upgrading to cf CLI v7](https://docs.cloudfoundry.org/cf-cli/v7.html) or [Upgrading to cf CLI v8](https://docs.cloudfoundry.org/cf-cli/v8.html).

* Run `cf v3-push APP-NAME`. See [v3-push - Cloud Foundry CLI](https://cli.cloudfoundry.org/en-US/cf/v3-push.html) for information about this function.
You can specify the schema version in the `versions` property of the manifest. This property is optional.
The only supported version is `1`. If not specified, the default value for the `versions` property is `1`.

### Add variables to a manifest
You can use variables to create app manifests with values shared across all applicable environments in combination with references to environment-specific differences defined in separate files.
To add variables to an app manifest:

1. Create a file called `vars.yml`.

2. Add attributes to your `vars.yml` file. For example:
```
instances: 2
memory: 1G
```

3. Add the variables to your app manifest file using the following format: `((VARIABLE-NAME))`. For example:
```

---
applications:

- name: test-app
instances: ((instances))
memory: ((memory))
buildpacks:

- go_buildpack
env:
GOPACKAGENAME: go_calls_ruby
command: go_calls_ruby
```

**Note**
You can also use variables for partial values. For example, you can specify `host` in your variables file and `- route: ((host)).env.com` in your manifest file.

4. Run:
```
cf push --vars-file /PATH/vars.yml
```
Where `PATH` is the path to the file you created.

### Minimize duplication with YAML anchors
Top-level attributes are deprecated in favor of YAML anchors. For more information, see [Top-level attributes](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest-attributes.html#top-level-attributes).
In manifests where multiple apps share settings or services, you might see duplicated content. While the manifests still work, duplication increases the risk of typographical errors, which cause deployments to fail.
You can declare shared configuration using a YAML anchor, to which the manifest refers in app declarations by using an alias.
```

---
defaults: &amp;defaults
buildpacks:

- staticfile_buildpack
memory: 1G
applications:

- name: bigapp
&lt;&lt;: *defaults

- name: smallapp
&lt;&lt;: *defaults
memory: 256M
```
In the example, manifest pushes two apps with the `staticfile` buildpack, `smallapp` and `bigapp`, with 256 M of memory for `smallapp` and 1 G of
memory for `bigapp`.

## App attributes
This section explains how to describe optional app attributes in manifests. You can also specify each of these attributes using a command line option. Command-line options override the manifest.

**Important**
In cf CLI v6, the route component attributes `domain`, `domains`, `host`,
`hosts`, and `no-hostname` are deprecated in favor of the `routes` attribute. In cf CLI v7, these attributes are
removed. For more information, see [domain, domains, host, hosts, and no-hostname](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest-attributes.html#route-attribute).

### buildpacks
You can refer to a buildpack by name in a manifest or a command-line option. The `cf buildpacks` command lists the buildpacks that you can use.

* **Custom buildpacks:** If your app requires a custom buildpack, you can use the `buildpacks` attribute to specify it in a number of ways:

+ By name: `BUILDPACK`.

+ By GitHub URL: `https://github.com/cloudfoundry/java-buildpack.git`.

+ By GitHub URL with a branch or tag: `https://github.com/cloudfoundry/java-buildpack.git#v3.3.0` for the `v3.3.0` tag.
```

---
...
buildpacks:

- buildpack_URL
```

* **Multiple buildpacks:** If you are using multiple buildpacks, you can provide an additional `-b` flag or add an additional value to your manifest:
```

---
...
buildpacks:

- buildpack_URL

- buildpack_URL
```

**Important**

+ This feature does not work with the deprecated `buildpack` attribute. For more information, see
[buildpack](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest-attributes.html#buildpack-deprecated).

+ You must specify multiple buildpacks in the correct order: the buildpack uses the app start command given by the final buildpack. For more information, see the [multi-buildpack](https://github.com/cloudfoundry/multi-buildpack#usageapp) repository on GitHub.
The `-b` command-line flag overrides this attribute.
For more information, see [Pushing an app with multiple buildpacks](https://docs.cloudfoundry.org/buildpacks/use-multiple-buildpacks.html).

### command
Some languages and frameworks require that you provide a custom command to start an app. To find out if you need to provide a custom start command, see [Buildpacks](https://docs.cloudfoundry.org/buildpacks/).
You can provide the custom start command in your app manifest or on the command line. For more information about how Cloud Foundry determines its default start command, see [Starting, restarting, and restaging apps](https://docs.cloudfoundry.org/devguide/deploy-apps/start-restart-restage.html).
To specify the custom start command in your app manifest, add it in the `command: START-COMMAND` format. For example:
```

---
...
command: bundle exec rake VERBOSE=true
```
The start command you specify becomes the default for your app. To return to using the original default start command set by your buildpack, you must explicitly set the attribute to `null`. For example:
```

---
...
command: null
```
On the command line, you can specify the custom start command by including the `-c` flag, similar to the following example:
```
cf push my-app -c "bundle exec rake VERBOSE=true"
```
The `-c` option with a value of `null` forces `cf push` to use the buildpack start command. For more information, see [Forcing cf push to use the Buildpack Start Command](https://docs.cloudfoundry.org/devguide/deploy-apps/start-restart-restage.html#revert) in *Starting, restarting, and restaging apps*.
If you override the start command for a Buildpack app, Linux uses `bash -c COMMAND` to run your app. If you override the start command for a Docker
app, Linux uses `sh -c COMMAND` to run your app. Because of this, if you override a start command, you must prefix `exec` to the final command in your custom composite start command.
An app must catch termination signals and clean up appropriately. Because of the way that shells manage process trees, the use of custom composite shell commands, particularly those that create child processes using `&`, `&&`, `||`, and so on, can prevent your app from receiving signals that are sent to the top-level bash process. For more information, see [Considerations for designing and running an app in the cloud](https://docs.cloudfoundry.org/devguide/deploy-apps/prepare-to-deploy.html#moving-apps).
To resolve this issue, you can use `exec` to replace the bash process with your own process.
For example:

* `bin/rake cf:on_first_instance db:migrate && bin/rails server -p $PORT -e $RAILS_ENV`: The process tree is bash -> ruby, so on graceful shutdown only the bash process receives the TERM signal, not the ruby process.

* `bin/rake cf:on_first_instance db:migrate && exec bin/rails server -p $PORT -e $RAILS_ENV`: Because of the `exec` prefix included on the final command, the Ruby process that is run by Rails takes over the bash process managing the execution of the composite command. The process tree is only Ruby, so the Ruby web server receives the TERM signal and can shut down gracefully for 10 seconds.
In more complex situations, like making a custom buildpack, you might want to use bash `trap`, `wait`, and background processes to manage your process tree and shut down apps gracefully. In most situations, however, a well-placed `exec` is sufficient.

### disk\_quota
The `disk_quota` attribute allocates the disk space for your app instance. This attribute requires a unit of measurement: `M`, `MB`, `G`, or `GB`, in either uppercase or lowercase.
For example:
```

---
...
disk_quota: 1024M
```
The `-k` command-line flag overrides this attribute.

### docker
If your app is contained in a Docker image, the `docker` attribute specifies it and an Docker user name (optional). This attribute is a combination of `push` options that include `--docker-image` and `--docker-username`.
For example:
```

---
...
docker:
image: docker-image-repository/docker-image-name
username: docker-user-name
```
The `--docker-image` or `-o` command-line flag overrides `docker.image`. The `--docker-username` command-line flag overrides `docker.username`.
The manifest attribute `docker.username` is optional. If it is used, the password must be provided in the environment variable `CF_DOCKER_PASSWORD`. If a Docker user name is specified, then a Docker image must also be specified.

**Important**
Using the `docker` attribute with the `buildpacks` or `path` attributes causes an error.

### health-check-type
The `health-check-type` attribute sets the `health_check_type` flag to either `port`, `process` or `http`. If you do not provide a `health-check-type` attribute, the default is `port`.
For example:
```

---
...
health-check-type: port
```
The `-u` command-line flag overrides this attribute.
In cf CLI v6, the value of `none` is deprecated in favor of `process`. In cf CLI v7, `none` is removed.

### health-check-http-endpoint
The `health-check-http-endpoint` attribute customizes the endpoint for the `http` health check type. If you do not provide a `health-check-http-endpoint` attribute, it uses endpoint `/`.
For example:
```

---
...
health-check-type: http
health-check-http-endpoint: /health
```

### health-check-invocation-timeout
The `health-check-invocation-timeout` attribute specifies the timeout in seconds for individual health check requests for http and port health checks. The default value is 1.
For example:
```

---
...
health-check-invocation-timeout: 30
```
To override this attribute, run:
```
cf set-health-check APP-NAME http --invocation-timeout 10
```
Where `APP-NAME` is the name of your app.
Within the manifest, the health check invocation timeout is controlled by the `health-check-invocation-timeout` attribute.

### health-check-interval
The `health-check-interval` attribute specifies the time in seconds between starting individual health check requests for HTTP and port health checks. The default value is 30.
For example:
```

---
...
health-check-interval: 15
```

### readiness-health-check-type
The `readiness-health-check-type` attribute sets the readiness health check type to `port`, `process` or `http`. If you do not provide a `readiness-health-check-type` attribute, the default is `process`.
For example:
```

---
...
readiness-health-check-type: port
```

### readiness-health-check-http-endpoint
The `readiness-health-check-http-endpoint` attribute customizes the endpoint for the `http` readiness health check types. If you do not provide a `readiness-health-check-http-endpoint` attribute, it uses endpoint `/`.
For example:
```

---
...
readiness-health-check-type: http
readiness-health-check-http-endpoint: /health
```

### readiness-health-check-invocation-timeout
The `readiness-health-check-invocation-timeout` attribute specifies the timeout in seconds for individual readiness health check requests for HTTP and port health checks. The default value is 1.
For example:
```

---
...
readiness-health-check-invocation-timeout: 30
```

### readiness-health-check-interval
The `readiness-health-check-interval` attribute specifies the amount of time in seconds between starting individual readiness health check requests for HTTP and port health checks.
For example:
```

---
...
readiness-health-check-interval: 15
```

### instances
The `instances` attribute configures the number of app instances.
For example:
```

---
...
instances: 2
```
The default number of instances is 1.
To ensure that platform maintenance does not interrupt your app, Cloud Foundry recommends running at least two instances.

### log-rate-limit-per-second
The `log-rate-limit-per-second` attribute specifies the log rate limit for all instances of an app. This attribute requires a unit of measurement: `B`, `K`, `KB`, `M`, `MB`, `G`, or `GB`, in either uppercase or lowercase.
For example:
```

---
...
log-rate-limit-per-second: 24KB
```
To configure each app instance to send an unlimited number of logs to Loggregator, specify `-1`.
The default log rate limit is 16 K. If you know that your app instances do not require the default log rate limit, you might want to
specify a smaller limit in your manifest to conserve quota space.
The `-l` command-line flag overrides this attribute.

### memory
The `memory` attribute specifies the memory limit for all instances of an app. This attribute requires a unit of measurement: `M`, `MB`, `G`, or `GB`, in either uppercase or lowercase.
For example:
```

---
...
memory: 1024M
```
The default memory limit is 1 G. If you know that your app instances do not require 1 G of memory, you might want to specify a smaller limit to conserve quota space.
The `-m` command-line flag overrides this attribute.

### metadata
The `metadata` attribute tags your apps with additional information. You can specify two types of metadata: `labels` and `annotations`. For more information,
see [Types of metadata](https://docs.cloudfoundry.org/adminguide/metadata.html#types) in *Using Metadata*.
For example:
```
metadata:
annotations:
contact: "bob@example.com jane@example.com"
labels:
sensitive: true
```
For more information about metadata, see [Using metadata](https://docs.cloudfoundry.org/adminguide/metadata.html).

### no-route

**Important**
If you use the `no-route` flag attribute in the manifest or the flag option, it overrides all route-related attributes.
By default, `cf push` assigns a route to every app. But, some apps process data while running in the background and must not be assigned routes.
You can use the `no-route` attribute with a value of `true` to prevent a route from being created for your app.
For example:
```

---
...
no-route: true
```
The `--no-route` command-line flag overrides this attribute.
In the Diego architecture, `no-route` skips creating and binding a route for the app, but does not specify which type of health check to perform. If your app does not listen on a port because it is a worker or a scheduler app, then it does not satisfy the port-based health check, and Cloud Foundry marks it as failed. To prevent this, deactivate the port-based health check by running:
```
cf set-health-check APP-NAME --process
```
Where `APP-NAME` is the name of your app.
To remove a route from an existing app:

1. Remove the route using the `cf unmap-route` command.

2. Push the app again with the `no-route: true` attribute in the manifest or the `--no-route` command line option.
For more information, see [Deploy multiple apps with one manifest](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest.html#multi-apps) in *Deploying with App Manifests*.

### path
The `path` attribute tells Cloud Foundry the directory location in which it can find your app. The directory specified as the `path`, either as an attribute or as a parameter on the command line, becomes the location where the buildpack `Detect` script runs.
For example:
```

---
...
path: /path/to/app/bits
```
The `-p` command-line flag overrides this attribute.
For more information, see [How cf push finds the app](https://docs.cloudfoundry.org/cf-cli/getting-started.html#find-app) in *Getting Started with the cf CLI*.

### processes

**Important**
This attribute is available with CAPI V3 only. To push a manifest that uses this attribute, do one of the following:

* Use cf CLI v7 or v8.
See [Upgrading to cf CLI v7](https://docs.cloudfoundry.org/cf-cli/v7.html) or [Upgrading to cf CLI v8](https://docs.cloudfoundry.org/cf-cli/v8.html).

* Run `cf v3-push APP-NAME`. See [v3-push - Cloud Foundry CLI](https://cli.cloudfoundry.org/en-US/cf/v3-push.html) for information about this function.
The `processes` attribute pushes apps that run multiple processes, such as a web app that has a UI process and a worker process.
For example:
```
processes:

- type: web
command: start-web.sh
disk_quota: 512M
health-check-http-endpoint: /healthcheck
health-check-type: http
health-check-invocation-timeout: 10
instances: 3
memory: 500M
timeout: 10

- type: worker
command: start-worker.sh
disk_quota: 1G
health-check-type: process
instances: 2
memory: 256M
timeout: 15
```
For detailed information about the process-level configuration, see the [CAPI
documentation](https://v3-apidocs.cloudfoundry.org/version/3.76.0/index.html#app-manifest).
For more information about pushing an app with multiple processes, see [Pushing an app with multiple processes](https://docs.cloudfoundry.org/devguide/multiple-processes.html).

### random-route
If you push your app without specifying any route-related CLI options or app manifest flags, the cf CLI attempts to generate a route based on the app name, which can cause collisions.
You can use the `random-route` attribute to generate a unique route and avoid name collisions. When you use `random-route`, the cf CLI generates one of the following:

* An HTTP route with a random host, if no value is specified for `host`

* A TCP route with an unused port number.
For example:
```

---
...
random-route: true
```
The following example use cases demonstrate when you might use the `random-route` attribute:

* You deploy the same app to multiple spaces for testing purposes. In this situation, you can use `random-route` to randomize routes declared with the route attribute in the app manifest.

* You use an app manifest for a classroom training exercise in which multiple users deploy the same app to the same space.
The `--random-route` command-line flag overrides this attribute.

### routes
The `routes` attribute in the manifest provides multiple HTTP and TCP routes. Each route for this app is created if it does not already exist.

**Important**
This attribute is a combination of `push` options that include `--hostname`, `-d`, and `--route-path` flags in v6. These flags are not supported in cf CLI v7, so the `routes` flag must be used.
You can specify the `protocol` attribute to configure which network protocol the route uses for app ingress traffic. This is optional. The available protocols are `http2`, `http1`, and `tcp`.

**Important**
The `protocol` route attribute is available only for Cloud Foundry deployments that use HTTP/2 routing. For information about configuring support for HTTP/2 in Cloud Foundry, see [Configuring HTTP/2 Support](https://docs.cloudfoundry.org/adminguide/supporting-http2.html).
For example:
```

---
...
routes:

- route: example.com
protocol: http2

- route: www.example.com/foo

- route: tcp-example.com:1234
```

#### Manifest attributes
If you use the `routes` attribute with the `host`, `hosts`, `domain`, `domains`, or `no-hostname` attributes, an error results.

#### push flag options
This attribute has unique interactions with different command-line options.
This table is updated for cf CLI v7; several of the flags were removed (`--route-path`, `-d`, `--hostname`, `--no-hostname`)
| Flag | Result |
| --- | --- |
| `--no-route` | All declared routes are ignored. In cf CLI v7, this flag no longer unbinds all existing routes associated with the app. |
| `--random-route` | Sets or overrides the `HOSTNAME` in all HTTP routes. Sets or overrides the `PORT` in all TCP routes. The `PORT` and
`HOSTNAME` are randomly generated. |

### sidecars

**Important**
This attribute is available with CAPI V3 only. To push a manifest that uses this attribute, do one of the following:

* Use cf CLI v7 or v8.
See [Upgrading to cf CLI v7](https://docs.cloudfoundry.org/cf-cli/v7.html) or [Upgrading to cf CLI v8](https://docs.cloudfoundry.org/cf-cli/v8.html).

* Run `cf v3-push APP-NAME`. See [v3-push - Cloud Foundry CLI](https://cli.cloudfoundry.org/en-US/cf/v3-push.html) for information about this function.
The `sidecars` attribute specifies additional processes to run in the same container as your app. Each sidecar must have values for `name`, `process_types`, and `command`, whereas `memory` is optional.
For example:
```
sidecars:

- name: authenticator
process_types: [ 'web', 'worker' ]
command: bundle exec run-authenticator
memory: 800M

- name: upcaser
process_types: [ 'worker' ]
command: ./tr-server
memory: 900M
```
For more information about sidecars, see [Pushing apps with sidecar processes](https://docs.cloudfoundry.org/devguide/sidecars.html).

### stack
The `stack` attribute specifies the stack to which your app deploys.
For example:
```

---
...
stack: cflinuxfs4
```
To see a list of available stacks, run `cf stacks`.
The `-s` command-line flag overrides this attribute.

### timeout
The `timeout` attribute defines the number of seconds that Cloud Foundry allocates for starting your app. It is related to the
`health-check-type` attribute.
For example:
```

---
...
timeout: 80
```
You can increase the timeout length for very large apps that require more time to start. The default for the `timeout` attribute is `60`, but you can set it to any value up to the Cloud Controller’s `cc.maximum_health_check_timeout` property.
`cc.maximum_health_check_timeout` defaults to the maximum of `180` seconds, but your Cloud Foundry operator can set it to a different value.
The `-t` command-line flag overrides this attribute.

**Important**
If you configure `timeout` with a value greater than
`cc.maximum_health_check_timeout`, the Cloud Controller reports a validation error with the maximum limit.

## Environment variables
The `env` block consists of a heading, then one or more environment variable/value pairs.
For example:
```

---
...
env:
RAILS_ENV: production
RACK_ENV: production
```
`cf push` deploys the app to a container on the server. The variables belong to the container environment.

**Important**

* You must name variables with alphanumeric characters and underscores. Non-conforming variable names might cause unpredictable behavior.

* Do not use user-provided environment variables for security sensitive information such as credentials, because they might unintentionally show up in cf CLI output and Cloud Controller logs. Use [user-provided service instances](https://docs.cloudfoundry.org/devguide/services/user-provided.html) instead. The system-provided environment variable [VCAP\_SERVICES](https://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html#VCAP-SERVICES) is properly redacted for user roles such as Space Supporter and in Cloud Controller log files.
While the app is running, you can edit environment variables using one of these methods:

* To view all variables, run:
```
cf env APP-NAME
```
Where `APP-NAME` is the name of your app.

* To set an individual variable, run:
```
cf set-env APP-NAME VARIABLE-NAME VARIABLE-VALUE
```
Where:

+ `APP-NAME` is the name of your app.

+ `VARIABLE-NAME` is the environment variable you want to set.

+ `VARIABLE-VALUE` is the value of the environment value.

* Removing an environment variable from the manifest YAML file is not sufficient to un-set it. To un-set an environment variable, run:
```
cf unset-env APP-NAME VARIABLE-NAME
```
Where:

+ `APP-NAME` is the name of your app.

+ `VARIABLE-NAME` is the environment variable you want to un-set.
Environment variables interact with manifests in the following ways:

* When you deploy an app for the first time, Cloud Foundry reads the variables described in the environment block of the manifest and adds them to the environment of the container where the app is staged, and the environment of the container where the app is deployed.

* When you stop and then restart an app, its environment variables persist.

## Services
Apps can bind to services such as databases, messaging, and key-value stores.
Apps are deployed into app spaces. An app can only bind to services instances that exist in the target app space before the app is deployed.
The `services` block consists of a heading and one or more service instance names.
The following ‘services’ attributes are allowed:
```

---
...
services:

- service-1

- name: service-2

- name: service-3
parameters:
key-1: value-1
key-2: [value-2, value-3]
key-3: ... any other kind of value ...

- name: service-4
binding_name: binding-1
```
The person who creates the service chooses the service instance names. These names can convey logical information, describe the nature of the service, or neither.
For example:
```

---
...
services:

- instance_ABC

- instance_XYZ
```
You can bind an app to a service instance by setting the `VCAP_SERVICES` environment variable. For more information, see [Bind a
service](https://docs.cloudfoundry.org/devguide/services/application-binding.html#bind) in *Delivering Service Credentials to an App*.

## Deprecated app manifest features
This section describes app manifest features that are deprecated in favor of other features.

****Caution****
Running `cf push app -f manifest.yml` fails if your manifest uses any of these deprecated features with the feature that replaces it.

### Top-level attributes
Previously, you could declare top-level attributes, which are also known as global attributes. For example, you can move an attribute above the `applications` block, where it need appear only once.
The following example demonstrates how this was used to manage duplicated settings:
```

---
domain: shared-domain.example.com
memory: 1G
instances: 1
services:

- clockwork-mysql
applications:

- name: springtock
host: tock09876
path: ./spring-music/build/libs/spring-music.war

- name: springtick
host: tick09875
path: ./spring-music/build/libs/spring-music.war
```
Top-level attributes are deprecated in favor of YAML aliases.
The following example demonstrates how to specify a shared configuration using a YAML anchor, which the manifest refers to in app declarations by using an alias:
```

---
defaults: &amp;defaults
buildpacks:

- staticfile_buildpack
memory: 1G
applications:

- name: bigapp
&lt;&lt;: *defaults

- name: smallapp
&lt;&lt;: *defaults
memory: 256M
```
When pushing the app, make explicit the attributes in each app’s declaration. To do this, assign the anchors and include the app-level attributes with YAML
aliases in each app declaration.

### domain, domains, host, hosts, and no-hostname attributes
These flags are removed in cf CLI v7.
Previously, you could specify routes by listing them all at once using the `routes` attribute, or by using their hosts and domains.
For example:
```

---
applications:

- name: webapp
host: www
domains:

- example.com

- example.io
```
The following route component attributes are deprecated:

* `domain`

* `domains`

* `host`

* `hosts`

* `no-hostname`
You can only specify routes using the `routes` attribute:
```

---
applications:

- name: webapp
routes:

- route: www.example.com/foo

- route: tcp.example.com:1234
```

### Inheritance
This feature is deprecated and replaced by variable substitution. For more information, see [Variable substitution](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest-attributes.html#variable-substitution).
With inheritance, child manifests inherited configurations from a parent manifest. The child manifests can use the inherited configurations as provided,
extend them, or override them.

### buildpack
The singular `buildpack` attribute is deprecated. It is replaced by `buildpacks`, which specifies multiple buildpacks. For more information, see
[buildpacks](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest-attributes.html#buildpacks).