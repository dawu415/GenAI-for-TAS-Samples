# PHP buildpacks in Cloud Foundry
For information about using and extending the PHP buildpack in Cloud Foundry,
see the [php-buildpack GitHub repository](https://github.com/cloudfoundry/php-buildpack).
You can find current information about this buildpack on the PHP buildpack
[release page](https://github.com/cloudfoundry/php-buildpack/releases) in
GitHub.
The buildpack uses a default PHP version specified in [.defaults/options.json](https://github.com/cloudfoundry/php-buildpack/blob/master/defaults/options.json) under the `PHP_VERSION` key.
To change the default version, specify the `PHP_VERSION` key in your app’s
`.bp-config/options.json` file.