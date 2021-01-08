$('#guess').on('click', function () {
    $('#result').text('通信中...');
    //インジケータ表示
    var indicator = $("#overlay").show()
    $.ajax({
        url: 'guess',
        type: 'GET',
        dataType: 'json',
        // フォーム要素の内容をハッシュ形式に変換
        data: $('form').serializeArray(),
        timeout: 5000,
    })
        .done(function (data) {
            //インジケータ除去
            indicator.hide();
            status = data.status
            if (status == '200') {
                result = '予測カテゴリー：' + data.category
            }
            else {
                result = data.message.url
            }
            $('#result').text(result);
        })
        .fail(function () {
            //インジケータ除去
            indicator.hide();
            // 通信失敗時の処理を記述
            $('#result').text('システムエラー');
        });
})