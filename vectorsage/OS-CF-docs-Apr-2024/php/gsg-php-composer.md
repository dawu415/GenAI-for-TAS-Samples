# Composer
Composer is activated when you supply a `composer.json` or `composer.lock` file.
A `composer.lock` is not required, but is strongly recommended for consistent deployments.
You can require dependencies for packages and extensions. Extensions must be prefixed with the standard `ext-`. If you reference an extension that is available to the buildpack, it automatically is installed. See the main [README](https://github.com/cloudfoundry/php-buildpack#supported-software) for a list of supported extensions.
The buildpack uses the version of PHP specified in your `composer.json` or `composer.lock` file. Composer settings override the version set in the `options.json` file.
The PHP buildpack supports a subset of the version formats supported by Composer. The buildpack supported formats are:
| Example | Expected Version |
| --- | --- |
| 5.3.\* | latest 5.4.x release (5.3 is not supported) |
| >=5.3 | latest 5.4.x release (5.3 is not supported) |
| 5.4.\* | latest 5.4.x release |
| >=5.4 | latest 5.4.x release |
| 5.5.\* | latest 5.5.x release |
| >=5.5 | latest 5.5.x release |
| 5.4.x | specific 5.4.x release that is listed |
| 5.5.x | specific 5.5.x release that is listed |

## Configuration
The buildpack runs with a set of default values for Composer.
You can adjust these values by adding a `.bp-config/options.json` file to your application and setting any of the following values in it.
| Variable | Explanation |
| --- | --- |
| COMPOSER\_VERSION | The version of Composer to use. It defaults to the latest bundled with the buildpack. |
| COMPOSER\_INSTALL\_OPTIONS | A list of options that should be passed to `composer install`. This defaults to `["--no-interaction", "--no-dev", "--no-progress"]`. The `--no-progress` option must be used due to the way the buildpack calls Composer. |
| COMPOSER\_VENDOR\_DIR | Allows you to override the default value used by the buildpack. This is passed through to Composer and instructs it where to create the `vendor` directory. Defaults to `{BUILD_DIR}/{LIBDIR}/vendor`. |
| COMPOSER\_BIN\_DIR | Allows you to override the default value used by the buildpack. This is passed through to Composer and instructs it where to place executables from packages. Defaults to `{BUILD_DIR}/php/bin`. |
| COMPOSER\_CACHE\_DIR | Allows you to override the default value used by the buildpack. This is passed through to Composer and instructs it where to place its cache files. Generally you should not change this value. The default is `{CACHE_DIR}/composer` which is a subdirectory of the cache folder passed in to the buildpack. Composer cache files are restored on subsequent application pushes. |
By default, the PHP buildpack uses the `composer.json` and `composer.lock` files that reside inside the root directory, or in the directory specified as `WEBDIR` in your `options.json`. If you have composer files inside your app, but not in the default directories, use a `COMPOSER_PATH` environment variable for your app to specify this custom location, relative to the app root directory. Note, that the `composer.json` and `composer.lock` files must be in the same directory.

## GitHub API request limits
Composer uses GitHub’s API to retrieve zip files for installation into the application folder. If you do not vendor dependencies before pushing an app, Composer can fetch dependencies during staging using the GitHub API.
GitHub’s API is request-limited. If you reach your daily allowance of API requests (typically 60), GitHub’s API returns a `403` error and staging fails.
There are two ways to avoid the request limit:

* Vendor dependencies before pushing your application.

* Supply a GitHub OAuth API token.

### Vendor dependencies
For vendor dependencies, you must run `composer install` before you push your application. You might also need to configure `COMPOSER_VENDOR_DIR` to “vendor”.

### Supply a GitHub token
Composer can use [GitHub API OAuth tokens](https://help.github.com/articles/creating-an-access-token-for-command-line-use/), which increase your request limit, typically to 5000 per day.
During staging, the buildpack looks for this token in the environment variable `COMPOSER_GITHUB_OAUTH_TOKEN`. If you supply a valid token, Composer uses it. This mechanism does not work if the token is invalid.
To supply the token, use one of the following methods:

* Run:
```
cf set-env YOUR_APP_NAME COMPOSER_GITHUB_OAUTH_TOKEN "OAUTH_TOKEN_VALUE"
```

* Add the token to the `env` block of your application manifest.

## Buildpack staging environment
Composer runs in the buildpack staging environment. Variables set with `cf set-env` or with [a manifest.yml ‘env’ block](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest.html#env-block) are visible to Composer.
For example:
```
$ cf push a_symfony_app --no-start
$ cf set-env a_symfony_app SYMFONY_ENV "prod"
$ cf start a_symfony_app
```
In this example, `a_symfony_app` is supplied with an environment variable, `SYMFONY_ENV`, which is visible to Composer and any scripts started by Composer.

### Non-configurable environment variables
User-assigned environment variables are applied to staging and runtime. Unfortunately, `LD_LIBRARY_PATH` and `PHPRC` must be different for staging and runtime. The buildpack takes care of setting these variables, which means user values for these variables are ignored.