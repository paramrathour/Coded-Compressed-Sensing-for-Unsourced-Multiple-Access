M = 11;
n = 2^M - 1;   % Codeword length
k = 23;        % Message length
nwords = 2^15;    % Number of words to encode
% Generate initial message and encode
msgTx = gf(randi([0 1], nwords, k));
enc = bchenc(msgTx, n, k);
A1 = enc.x';
save('BCH_matrix.mat', 'A1');

nwords = 1;    % Number of words to encode
% Generate initial message and encode
msgTx = gf(randi([0 1], nwords, k));
enc = bchenc(msgTx, n, k);

% Initialize matrix to store encoded messages
enc_matrix_size = [2047, 2^15];
encoded_messages = gf(zeros(enc_matrix_size), 1);
encoded_messages(:, 1) = enc';
entries = 1;
while entries <= 2^15
    % Generate new message and encode
    msgTx = gf(randi([0 1], nwords, k));
    enc = bchenc(msgTx, n, k);

    % Check if enc has occurred previously
    is_duplicate = any(all(encoded_messages(:, 1:entries) == enc', 1)) | any(all(encoded_messages(:, 1:entries) == enc'+1, 1));

    if ~is_duplicate
        entries = entries + 1;
        encoded_messages(:, entries) = enc';
    end
end
A2 = encoded_messages.x;
save('BCH_matrix.mat', 'A1', 'A2');
% Display the number of distinct encoded messages generated
disp(['Number of distinct encoded messages: ', num2str(entries-1)]);