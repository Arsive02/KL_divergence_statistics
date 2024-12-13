const References = () => {
  const references = [
    {
      id: 'blei2017variational',
      citation: 'Blei, D.M., Kucukelbir, A., McAuliffe, J.D., 2017. Variational inference: A review for statisticians. Journal of the American Statistical Association 112(518), 859-877.'
    },
    {
      id: 'kingma2013auto',
      citation: 'Kingma, D.P., 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.'
    },
    {
      id: 'bishop2006pattern',
      citation: 'Bishop, C.M., 2006. Pattern Recognition and Machine Learning. Springer, New York.'
    },
    {
      id: 'thomas2006elements',
      citation: 'Thomas, M.T.C.A.J., Joy, A.T., 2006. Elements of Information Theory. Wiley-Interscience.'
    },
    {
      id: 'cover2012elements',
      citation: 'Cover, T.M., Thomas, J.A., 2012. Elements of Information Theory. Wiley.'
    },
    {
      id: 'murphy2012probabilistic',
      citation: 'Murphy, K.P., 2012. Machine Learning: A Probabilistic Perspective. MIT Press.'
    },
    {
      id: 'gao2017distributional',
      citation: 'Gao, W., Verdu, S., 2017. Distributional Transformations for Binary Hypothesis Testing. IEEE Transactions on Information Theory 63(1), 242-271.'
    },
    {
      id: 'moreno2003kullback',
      citation: 'Moreno, P., Ho, P., Vasconcelos, N., 2003. A Kullback-Leibler divergence based kernel for SVM classification in multimedia applications. Advances in Neural Information Processing Systems 16.'
    },
    {
      id: 'chen2016variational',
      citation: 'Chen, X., Kingma, D.P., Salimans, T., Duan, Y., Dhariwal, P., Schulman, J., Sutskever, I., Abbeel, P., 2016. Variational lossy autoencoder. arXiv preprint arXiv:1611.02731.'
    },
    {
      id: 'wang2019drifted',
      citation: 'Wang, X., Kang, Q., An, J., Zhou, M., 2019. Drifted Twitter spam classification using multiscale detection test on KL divergence. IEEE Access 7, 108384-108394.'
    },
    {
      id: 'painsky2018universality',
      citation: 'Painsky, A., Wornell, G., 2018. On the universality of the logistic loss function. In: 2018 IEEE International Symposium on Information Theory (ISIT), 936-940.'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto py-8 px-4">
        <h2 className="text-2xl font-semibold mb-8">References</h2>
        
        <div className="bg-white rounded-lg shadow-sm p-8">
          <div className="space-y-4">
            {references.map((ref, index) => (
              <div 
                key={ref.id}
                className="pl-8 -indent-8"
                id={ref.id}
              >
                <p className="text-gray-800">
                  [{index + 1}] {ref.citation}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-8 text-sm text-gray-600">
          <p>Total references: {references.length}</p>
        </div>
      </div>
    </div>
  );
};

export default References;