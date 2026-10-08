# Gap analysis and future recommendations

Running list of gaps and recommendations found while building the PoC. It feeds the
"Gap Analysis" and "Future Recommendations" sections of the handover presentation.

## Retrieval quality

### Choose an embedding model built for cross-language retrieval

- **Finding:** The source document is the Italian edition of the AWS Serverless Applications
  Lens (December 2019), while engineers are expected to ask questions in English. The PoC uses
  Amazon Titan Text Embeddings V2, which AWS documents as optimised for English, and AWS states
  that cross-language queries "will return sub-optimal results".
- **Why Titan V2 was kept for the PoC:** it is an Amazon first-party model (no AWS Marketplace
  subscription, so deployment to a clean account stays fully automated), it runs in-region in
  eu-central-1, and it has the lowest cost.
- **Recommendation:** before production, evaluate a model designed for multilingual and
  cross-language retrieval, such as Cohere Embed v4 (available from eu-central-1 through the EU
  cross-region inference profile `eu.cohere.embed-v4:0`). Compare it with Titan V2 on the PoC's
  evaluation question set before deciding.
- **Things to plan for when switching:**
  - Cohere is billed through AWS Marketplace, so the first invocation in an account needs
    Marketplace permissions.
  - Changing the embedding model means re-embedding all documents into a new vector index.
- **Note:** if the client standardises on the English edition of the document, Titan V2 is a
  good fit and this item can be dropped.

## Security

### End-user authentication with Microsoft Entra ID

- **Gap:** authentication is out of scope for the PoC. The query endpoint (Lambda Function URL)
  uses IAM authentication, so only callers with AWS credentials in the account can use it.
- **Recommendation:**
  - **Engineers using the CLI:** federate Microsoft Entra ID with AWS IAM Identity Center. Users
    sign in with `aws sso login` using their Microsoft account. No application change is needed.
  - **Web UI for users without AWS access:** put Amazon API Gateway (HTTP API) in front of the
    query function with a JWT authorizer that trusts the client's Entra ID tenant. This needs an
    Entra app registration, created by a tenant administrator.
