# S1 AFI build artifacts

The text logs, timing reports, and build metadata in this directory are the
reviewable evidence retained in ordinary Git. Six generated AWS HDK products
are intentionally excluded because each is 165--236 MiB and ordinary GitHub
Git objects are limited to 100 MiB. These three builds missed post-route timing;
the omitted files are archival implementation products, not claim-bearing
timing-closed bitstreams.

## Retrieve from S3

The three verified object prefixes are:

```text
s3://tiki-sweep-739966866637/s1afibuild-20260623T190446Z/results/to_aws/
s3://tiki-sweep-739966866637/s1afibuild-20260624T053140Z/results/to_aws/
s3://tiki-sweep-739966866637/s1afibuild-20260624T164218Z/results/to_aws/
```

From the repository root, using an AWS profile with access to that bucket:

```bash
mkdir -p results/fpga_silicon/s1_afibuild/to_aws
aws s3 cp \
  s3://tiki-sweep-739966866637/s1afibuild-20260623T190446Z/results/to_aws/ \
  results/fpga_silicon/s1_afibuild/to_aws/ \
  --recursive --profile <aws-profile> --region us-east-1
aws s3 cp \
  s3://tiki-sweep-739966866637/s1afibuild-20260624T053140Z/results/to_aws/ \
  results/fpga_silicon/s1_afibuild/to_aws/ \
  --recursive --profile <aws-profile> --region us-east-1
aws s3 cp \
  s3://tiki-sweep-739966866637/s1afibuild-20260624T164218Z/results/to_aws/ \
  results/fpga_silicon/s1_afibuild/to_aws/ \
  --recursive --profile <aws-profile> --region us-east-1
shasum -a 256 -c results/fpga_silicon/s1_afibuild/to_aws.SHA256SUMS
```

The fetched directory is ignored by Git. Do not add these products to an
ordinary source commit or paper artifact; publish them in S3 or another
large-object archive and retain the checksum manifest instead.

## Recover from the local exact archive

The workstation used for the 2026-08-31 GitHub publication keeps the unfiltered
objects reachable at this custom, local-only ref:

```text
refs/local-archive/tikitorch/pre-github-filter-2026-08-31
```

If that object database is available, materialize the directory from the
repository root with:

```bash
git archive \
  refs/local-archive/tikitorch/pre-github-filter-2026-08-31 \
  results/fpga_silicon/s1_afibuild/to_aws \
  | tar -xf -
shasum -a 256 -c results/fpga_silicon/s1_afibuild/to_aws.SHA256SUMS
```

That custom ref is deliberately not under `refs/heads` or `refs/tags`, so a
normal branch/tag push cannot upload the oversized historical blobs.

## Reproduce a new build

[`scripts/run_s1_afi_build.sh`](../../../scripts/run_s1_afi_build.sh) packages a
clean Git commit, launches the AWS FPGA Developer AMI build, uploads generated
products to S3, and supports `status` and `fetch` commands. Use a new run tag
and archive the resulting toolchain, `aws-fpga` commit, source commit, build
arguments, reports, S3 prefix, and hashes. A new build must pass the paper's
timing and provenance gates before it becomes claim-bearing evidence.
