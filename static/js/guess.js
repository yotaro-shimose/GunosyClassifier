$('#guess').on('click', function () {
    $('#result').text('通信中...');
    $.ajax({
        url: 'guess',
        type: 'GET',
        dataType: 'json',
        // フォーム要素の内容をハッシュ形式に変換
        data: $('form').serializeArray(),
        timeout: 5000,
    })
        .done(function (data) {
            status = data.status
            console.log(data)
            if (status == '200') {
                result = '予測カテゴリー：' + data.category
            }
            else {
                result = data.message.url
            }
            $('#result').text(result);
        })
        .fail(function () {
            // 通信失敗時の処理を記述
            $('#result').text('システムエラー');

        });
})